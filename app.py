import os
import json
import re
import asyncio
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from literature_search import LiteratureSearcher
from pdf_processor import extract_text_from_pdf_base64
from rag_utils import SimpleRAG
import numpy as np


load_dotenv()

app = FastAPI(title="AI Scientist Challenge Submission")

app.mount("/static", StaticFiles(directory="static"), name="static")

client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

reasoning_client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

embedding_base_url = os.getenv("SCI_EMBEDDING_BASE_URL")
embedding_api_key = os.getenv("SCI_EMBEDDING_API_KEY")
if embedding_base_url and embedding_api_key:
    embedding_client = AsyncOpenAI(
        base_url=embedding_base_url,
        api_key=embedding_api_key
    )
else:
    embedding_client = client


def detect_language(text: str) -> str:
    """
    检测文本的主要语言

    Args:
        text: 输入文本

    Returns:
        'zh' 表示中文，'en' 表示英文
    """
    if not text:
        return 'en'

    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))

    if total_chars == 0:
        return 'en'

    # 如果中文字符占比超过30%，认为是中文
    if chinese_chars / total_chars > 0.3:
        return 'zh'

    return 'en'


async def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
        
    Returns:
        嵌入向量
    """
    try:
        embedding_model = os.getenv("SCI_EMBEDDING_MODEL", "text-embedding-v4")
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量错误: {e}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        相似度值（0-1之间）
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses standard LLM model
    
    Request body:
    {
        "query": "What are the latest advances in transformer models?"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[literature_review] Received query: {query}")
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        language = detect_language(query)
        print(f"[literature_review] Detected language: {language}")

        async def generate():
            try:
                # 步骤1: 搜索相关文献
                papers_context = ""
                papers_references = []
                try:
                    searcher = LiteratureSearcher()
                    try:
                        # 并行搜索最新文献和高引用文献
                        latest_papers_task = searcher.search_and_extract(
                            query=query,
                            max_results=10,
                            extract_pdf=False,
                            sort_by='date'
                        )
                        
                        cited_papers_task = searcher.search_and_extract(
                            query=query,
                            max_results=10,
                            extract_pdf=False,
                            sort_by='citations'
                        )
                        
                        latest_papers, cited_papers = await asyncio.gather(
                            latest_papers_task,
                            cited_papers_task,
                            return_exceptions=True
                        )
                        
                        if isinstance(latest_papers, Exception):
                            print(f"[literature_review] 最新文献搜索错误: {latest_papers}")
                            latest_papers = []
                        if isinstance(cited_papers, Exception):
                            print(f"[literature_review] 高引用文献搜索错误: {cited_papers}")
                            cited_papers = []
                        
                        all_papers = []
                        seen_titles = set()
                        
                        for paper in latest_papers:
                            title = paper.get('title', '').lower().strip()
                            if title and title not in seen_titles:
                                all_papers.append(paper)
                                seen_titles.add(title)
                        
                        for paper in cited_papers:
                            title = paper.get('title', '').lower().strip()
                            if title and title not in seen_titles and len(all_papers) < 10:
                                all_papers.append(paper)
                                seen_titles.add(title)
                        
                        papers = all_papers
                        
                        print(f"[literature_review] 找到 {len(latest_papers)} 篇最新文献和 {len(cited_papers)} 篇高引用文献，合并后共 {len(papers)} 篇")
                        
                        if papers:
                            papers_list = []
                            for paper in papers:
                                if not paper.get('abstract') or not paper.get('abstract').strip():
                                    continue
                                
                                authors = paper.get('authors', [])[:3] if paper.get('authors') else []
                                published = paper.get('published', '')
                                year = ''
                                if published:
                                    year_match = re.search(r'(\d{4})', published)
                                    if year_match:
                                        year = year_match.group(1)
                                
                                author_surname = ''
                                if authors:
                                    first_author = authors[0]
                                    if ',' in first_author:
                                        author_surname = first_author.split(',')[0].strip()
                                    else:
                                        author_surname = first_author.split()[-1]
                                else:
                                    author_surname = 'Unknown'
                                
                                if year:
                                    if len(authors) > 1:
                                        citation_label = f"{author_surname} et al. ({year})"
                                    else:
                                        citation_label = f"{author_surname} ({year})"
                                else:
                                    citation_label = author_surname
                                
                                if language == 'zh':
                                    paper_info = f"{citation_label}: {paper.get('title', '未知')}"
                                    authors_str = ""
                                    if authors:
                                        authors_str = ', '.join(authors)
                                        paper_info += f"\n   作者: {authors_str}"
                                    if published:
                                        paper_info += f"\n   发表年份: {published}"
                                    paper_info += f"\n   摘要: {paper.get('abstract', '')[:300]}"
                                else:
                                    paper_info = f"{citation_label}: {paper.get('title', 'Unknown')}"
                                    authors_str = ""
                                    if authors:
                                        authors_str = ', '.join(authors)
                                        paper_info += f"\n   Authors: {authors_str}"
                                    if published:
                                        paper_info += f"\n   Published: {published}"
                                    paper_info += f"\n   Abstract: {paper.get('abstract', '')[:300]}"
                                
                                papers_list.append(paper_info)
                                
                                ref_info = {
                                    'author_surname': author_surname,
                                    'title': paper.get('title', ''),
                                    'authors': authors,
                                    'published': published,
                                    'year': year,
                                    'url': paper.get('url', ''),
                                    'doi': paper.get('doi', ''),
                                    'multiple_authors': len(authors) > 1
                                }
                                papers_references.append(ref_info)
                                
                                if len(papers_references) >= 10:
                                    break
                            
                            if papers_list:
                                if language == 'zh':
                                    papers_context = "\n\n相关文献:\n" + "\n\n".join(papers_list)
                                else:
                                    papers_context = "\n\nRelevant Papers:\n" + "\n\n".join(papers_list)
                    finally:
                        await searcher.close()
                except Exception as e:
                    print(f"[literature_review] 文献搜索警告: {e}，继续生成综述")

                # 步骤2: 根据语言准备prompt进行文献综述
                if language == 'zh':
                    citation_instruction = """
在综述中，请使用作者年份格式引用上述文献。可以使用两种格式：
1. 括号内引用格式：(Smith, 2023) 或 (Smith et al., 2023)，例如："transformer模型在自然语言处理中取得了显著进展 (Smith et al., 2023)。"
2. 作者在前格式：Smith (2023) 或 Smith et al. (2023)，例如："Smith et al. (2023) 提出了..." 或 "根据 Smith (2023) 的研究..."

请在适当的地方插入引用，根据语境选择合适的格式。
"""
                    if not papers_context:
                        citation_instruction = ""
                    
                    prompt = f"""请对以下研究主题进行全面的文献综述：

{query}

{papers_context}

{citation_instruction}

请提供一份结构化的文献综述，使用Markdown格式，包含以下部分：

1. **研究背景**
   - 该研究领域的历史发展和演进过程
   - 该领域的重要性和现实意义
   - 研究问题产生的背景和动机

2. **关键主题**
   - 该领域的核心研究主题和概念框架
   - 主要理论观点和方法论
   - 关键技术的原理和应用

3. **研究现状**
   - 当前研究的主要进展和突破
   - 不同研究方向的发展状况
   - 重要研究成果和发现

4. **研究趋势**
   - 最新发展和前沿技术
   - 新兴的研究方向和方法
   - 技术演进趋势

5. **研究空白**
   - 当前研究的局限性
   - 尚未解决的问题和挑战
   - 潜在的改进方向

6. **结论**
   - 总结该领域的研究现状和主要贡献
   - 概述关键研究发现和进展
   - 指出未来研究的重要方向和机会

{citation_instruction}

写作要求：
- 确保综述全面、准确、有深度，体现对领域的深入理解
- 在适当的地方使用作者年份格式的文献引用，如 (Smith, 2023)、(Smith et al., 2023)、Smith (2023) 或 Smith et al. (2023)
- 保持逻辑清晰，各部分之间衔接自然
- 结论部分应综合前述内容，提出有价值的见解"""
                else:
                    citation_instruction = """
In your review, please cite the above papers using author-year format. You can use two formats:
1. Parenthetical citation: (Smith, 2023) or (Smith et al., 2023), for example: "transformer models have achieved significant progress in natural language processing (Smith et al., 2023)."
2. Narrative citation: Smith (2023) or Smith et al. (2023), for example: "Smith et al. (2023) proposed..." or "According to Smith (2023)..."

Insert citations at appropriate places and choose the format that fits the context.
"""
                    if not papers_context:
                        citation_instruction = ""
                    
                    prompt = f"""Conduct a comprehensive literature review on the following topic:

{query}

{papers_context}

{citation_instruction}

Please provide a structured literature review in Markdown format covering:

1. **Background**
   - Historical development and evolution of the research field
   - Significance and real-world importance of the area
   - Context and motivation for the research questions

2. **Key Themes**
   - Core research themes and conceptual frameworks
   - Main theoretical perspectives and methodologies
   - Principles and applications of key technologies

3. **Current State**
   - Major advances and breakthroughs in current research
   - Development status of different research directions
   - Important research outcomes and findings

4. **Research Trends**
   - Latest developments and cutting-edge technologies
   - Emerging research directions and methods
   - Technological evolution trends

5. **Research Gaps**
   - Limitations of current research
   - Unsolved problems and challenges
   - Potential areas for improvement

6. **Conclusion**
   - Summarize the current state of research and major contributions in the field
   - Outline key research findings and progress
   - Identify important future research directions and opportunities

{citation_instruction}

Writing Requirements:
- Ensure the review is thorough, accurate, and insightful, demonstrating deep understanding of the field
- Use author-year format citations at appropriate places, such as (Smith, 2023), (Smith et al., 2023), Smith (2023), or Smith et al. (2023)
- Maintain clear logic and natural transitions between sections
- The conclusion should synthesize the preceding content and provide valuable insights"""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                full_text = ""
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            full_text += delta_content
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                # 添加参考文献列表
                if papers_references:
                    print(f"[literature_review] 开始检测引用，共有 {len(papers_references)} 篇参考文献")
                    
                    cited_refs = []
                    
                    if full_text:
                        print(f"[literature_review] 生成文本长度: {len(full_text)} 字符")
                        
                        for ref in papers_references:
                            author_surname = ref.get('author_surname', 'Unknown')
                            year = ref.get('year', '')
                            multiple_authors = ref.get('multiple_authors', False)
                            title = ref.get('title', '').lower()
                            
                            # 构建更全面的匹配模式
                            patterns = []
                            if year:
                                if multiple_authors:
                                    # 多种 et al. 格式
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?,? {year})")
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?, {year})")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? \\({year}\\)")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? {year}")
                                    # 支持中文环境下的格式
                                    patterns.append(f"（{re.escape(author_surname)} et al\\.?,? {year}）")
                                    patterns.append(f"（{re.escape(author_surname)}等,? {year}）")
                                else:
                                    # 单作者格式
                                    patterns.append(f"({re.escape(author_surname)},? {year})")
                                    patterns.append(f"({re.escape(author_surname)} {year})")
                                    patterns.append(f"{re.escape(author_surname)} \\({year}\\)")
                                    patterns.append(f"{re.escape(author_surname)} {year}")
                                    # 支持中文环境下的格式
                                    patterns.append(f"（{re.escape(author_surname)},? {year}）")
                                    patterns.append(f"（{re.escape(author_surname)} {year}）")
                            else:
                                # 没有年份的情况
                                if multiple_authors:
                                    patterns.append(f"{re.escape(author_surname)} et al\\.")
                                    patterns.append(f"{re.escape(author_surname)}等")
                                else:
                                    patterns.append(re.escape(author_surname))
                            
                            cited = False
                            # 先尝试模式匹配
                            for pattern in patterns:
                                if re.search(pattern, full_text, re.IGNORECASE):
                                    cited = True
                                    print(f"[literature_review] 找到引用: {author_surname} ({year})")
                                    break
                            
                            # 如果模式匹配失败，尝试标题匹配（作为备用方案）
                            if not cited and title:
                                # 提取标题的关键词（前几个词）
                                # 处理中文标题：按字符分割
                                if any('\u4e00' <= char <= '\u9fff' for char in title):
                                    # 中文标题：取前10个字符作为关键词
                                    title_keywords = title[:10] if len(title) >= 10 else title
                                    if len(title_keywords) > 3 and title_keywords in full_text.lower():
                                        cited = True
                                        print(f"[literature_review] 通过标题关键词找到引用: {author_surname} ({year})")
                                else:
                                    # 英文标题：取前5个词
                                    title_words = title.split()[:5]
                                    if title_words:
                                        # 过滤掉太短的词（如 a, an, the, in, of 等）
                                        important_words = [w for w in title_words if len(w) > 3]
                                        if important_words:
                                            # 检查至少3个重要词是否出现在文本中
                                            found_count = sum(1 for word in important_words if re.search(r'\b' + re.escape(word) + r'\b', full_text, re.IGNORECASE))
                                            if found_count >= min(3, len(important_words)):
                                                cited = True
                                                print(f"[literature_review] 通过标题关键词找到引用: {author_surname} ({year})")
                            
                            if cited:
                                cited_refs.append(ref)
                    else:
                        print(f"[literature_review] 警告: full_text 为空，无法检测引用")
                        cited_refs = papers_references
                    
                    print(f"[literature_review] 检测到 {len(cited_refs)} 篇被引用的文献")
                    
                    # 如果检测到的引用数量太少（少于提供的文献的50%），可能是检测逻辑有问题
                    # 此时将所有提供的文献都包含进来（保守策略，确保完整性）
                    if len(cited_refs) < len(papers_references) * 0.5 and len(papers_references) > 0:
                        print(f"[literature_review] 警告: 检测到的引用数量较少（{len(cited_refs)}/{len(papers_references)}），可能是匹配模式问题")
                        print(f"[literature_review] 将包含所有提供的文献以确保完整性")
                        cited_refs = papers_references
                    elif len(cited_refs) == 0 and len(papers_references) > 0:
                        # 如果完全没有检测到引用，也包含所有文献
                        print(f"[literature_review] 警告: 未检测到任何引用，将包含所有提供的文献")
                        cited_refs = papers_references
                    
                    if cited_refs:
                        def get_sort_key(ref):
                            surname = ref.get('author_surname', 'Unknown').lower()
                            year = ref.get('year', '')
                            return (surname, year)
                        
                        sorted_refs = sorted(cited_refs, key=get_sort_key)
                        
                        if language == 'zh':
                            references_text = "\n\n## 参考文献\n\n"
                        else:
                            references_text = "\n\n## References\n\n"
                        
                        for ref in sorted_refs:
                            author_surname = ref.get('author_surname', 'Unknown')
                            year = ref.get('year', '')
                            title = ref.get('title', '')
                            published = ref.get('published', '')
                            doi = ref.get('doi', '')
                            url = ref.get('url', '')
                            
                            # 处理published字段，提取年份或格式化日期
                            # 如果published是时间戳格式（如 2025-11-13T18:59:53Z），提取年份
                            cleaned_published = None
                            display_year = None
                            
                            if published:
                                # 检查是否是时间戳格式（包含T或Z，或ISO 8601格式）
                                if 'T' in published or 'Z' in published:
                                    # ISO 8601时间戳格式，提取年份
                                    year_match = re.search(r'(\d{4})', published)
                                    if year_match:
                                        display_year = year_match.group(1)
                                    cleaned_published = None
                                # 检查是否是日期格式（YYYY-MM-DD）
                                elif re.match(r'^\d{4}-\d{2}-\d{2}', published):
                                    # 日期格式，只提取年份
                                    year_match = re.search(r'(\d{4})', published)
                                    if year_match:
                                        display_year = year_match.group(1)
                                    cleaned_published = None
                                # 检查是否是年月格式（YYYY-MM）
                                elif re.match(r'^\d{4}-\d{2}$', published):
                                    # 年月格式，保持原样但也可以作为年份使用
                                    year_match = re.search(r'(\d{4})', published)
                                    if year_match:
                                        display_year = year_match.group(1)
                                    cleaned_published = None
                                # 检查是否是纯年份（YYYY）
                                elif re.match(r'^\d{4}$', published):
                                    display_year = published
                                    cleaned_published = None
                                else:
                                    # 可能是期刊名或其他发表信息，保留原样
                                    cleaned_published = published
                                    # 如果其中包含年份，提取出来
                                    year_match = re.search(r'(\d{4})', published)
                                    if year_match:
                                        extracted_year = year_match.group(1)
                                        # 如果year字段存在且与提取的年份相同，使用year
                                        if year and year == extracted_year:
                                            display_year = year
                                            cleaned_published = None
                                        # 如果没有year字段，使用提取的年份
                                        elif not year:
                                            display_year = extracted_year
                                            cleaned_published = None
                            
                            # 如果year字段存在且没有被使用，优先使用year
                            if year and not display_year:
                                display_year = year
                            elif year and display_year and year != display_year:
                                # 如果year和display_year不同，使用year（更准确）
                                display_year = year
                            
                            # 格式化作者（Elsevier格式：姓, 名首字母缩写）
                            if ref['authors']:
                                authors_list = ref['authors']
                                if len(authors_list) == 1:
                                    # 单作者：姓, 名首字母缩写
                                    author_parts = authors_list[0].split()
                                    if len(author_parts) >= 2:
                                        authors_str = f"{author_parts[-1]}, {'. '.join([p[0].upper() for p in author_parts[:-1]])}."
                                    else:
                                        authors_str = authors_list[0]
                                elif len(authors_list) == 2:
                                    # 两作者：姓1, 名1缩写, 姓2, 名2缩写（用逗号连接，最后用and）
                                    author1_parts = authors_list[0].split()
                                    author2_parts = authors_list[1].split()
                                    if len(author1_parts) >= 2 and len(author2_parts) >= 2:
                                        author1 = f"{author1_parts[-1]}, {'. '.join([p[0].upper() for p in author1_parts[:-1]])}."
                                        author2 = f"{author2_parts[-1]}, {'. '.join([p[0].upper() for p in author2_parts[:-1]])}."
                                        authors_str = f"{author1}, {author2}"
                                    else:
                                        authors_str = f"{authors_list[0]}, {authors_list[1]}"
                                elif len(authors_list) <= 10:
                                    # 3-10个作者：全部列出，用逗号分隔，最后一个用and连接
                                    formatted_authors = []
                                    for author in authors_list:
                                        author_parts = author.split()
                                        if len(author_parts) >= 2:
                                            formatted_author = f"{author_parts[-1]}, {'. '.join([p[0].upper() for p in author_parts[:-1]])}."
                                            formatted_authors.append(formatted_author)
                                        else:
                                            formatted_authors.append(author)
                                    # 最后一个用and连接
                                    if len(formatted_authors) > 1:
                                        authors_str = ", ".join(formatted_authors[:-1]) + ", and " + formatted_authors[-1]
                                    else:
                                        authors_str = formatted_authors[0]
                                else:
                                    # 超过10个作者：前10个 + et al.
                                    formatted_authors = []
                                    for author in authors_list[:10]:
                                        author_parts = author.split()
                                        if len(author_parts) >= 2:
                                            formatted_author = f"{author_parts[-1]}, {'. '.join([p[0].upper() for p in author_parts[:-1]])}."
                                            formatted_authors.append(formatted_author)
                                        else:
                                            formatted_authors.append(author)
                                    authors_str = ", ".join(formatted_authors) + ", et al."
                            else:
                                authors_str = 'Unknown Authors'
                            
                            # Elsevier格式：作者. 标题. 期刊名, 年份, 卷(期): 页码. DOI: xxx
                            # 简化格式（无卷期页码时）：作者. 标题. 期刊名/发表信息, 年份. DOI: xxx
                            if language == 'zh':
                                # 中文格式（Elsevier风格）
                                ref_line = f"{authors_str}. {title}."
                                
                                # 添加发表信息（期刊名等，但不包括年份）
                                if cleaned_published and not display_year:
                                    # 如果cleaned_published不是年份，可能是期刊名等信息
                                    ref_line += f" {cleaned_published}"
                                
                                # 添加年份
                                if display_year:
                                    if cleaned_published and not display_year:
                                        ref_line += f", {display_year}"
                                    else:
                                        ref_line += f" {display_year}"
                                
                                if doi:
                                    # 标准化DOI（移除URL前缀）
                                    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
                                    ref_line += f" DOI: {clean_doi}"
                                elif url:
                                    ref_line += f" {url}"
                            else:
                                # 英文格式（Elsevier风格）
                                # 格式：Author(s). Title. Journal/Publication, Year. DOI: xxx
                                ref_line = f"{authors_str} {title}."
                                
                                # 添加发表信息（期刊名等，但不包括年份）
                                if cleaned_published and not display_year:
                                    # 如果cleaned_published不是年份，可能是期刊名等信息
                                    ref_line += f" {cleaned_published}"
                                
                                # 添加年份
                                if display_year:
                                    if cleaned_published and not display_year:
                                        ref_line += f", {display_year}."
                                    else:
                                        ref_line += f" {display_year}."
                                else:
                                    ref_line += "."
                                
                                if doi:
                                    # 标准化DOI（移除URL前缀）
                                    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
                                    ref_line += f" DOI: {clean_doi}"
                                elif url:
                                    ref_line += f" {url}"
                            
                            references_text += ref_line + "\n\n"
                        
                        print(f"[literature_review] 准备发送参考文献列表，共 {len(sorted_refs)} 篇")
                        
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": references_text
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                    else:
                        print(f"[literature_review] 警告: 未找到任何被引用的文献")
                else:
                    print(f"[literature_review] 警告: papers_references 为空")

                yield "data: [DONE]\n\n"
                
            except asyncio.TimeoutError:
                error_data = {
                    "object": "error",
                    "message": "Request timeout (15 minutes limit)"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[literature_review] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    Paper Q&A endpoint - uses reasoning model with PDF content
    
    Request body:
    {
        "query": "Please carefully analyze and explain the reinforcement learning training methods used in this article.",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )
        
        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_qa] Received query: {query}")
        print(f"[paper_qa] Using reasoning model: {os.getenv('SCI_LLM_REASONING_MODEL')}")

        language = detect_language(query)
        print(f"[paper_qa] Detected language: {language}")

        async def generate():
            try:
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    error_data = {
                        "object": "error",
                        "message": error_msg
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                if len(pdf_text) > 8000:
                    rag = SimpleRAG()
                    relevant_chunks = await rag.retrieve_relevant_chunks(query, pdf_text, top_k=5)
                    context = "\n\n".join(relevant_chunks)
                    paper_content = context
                else:
                    paper_content = pdf_text
                
                if language == 'zh':
                    prompt = f"""请基于以下论文内容回答问题。

论文内容:

{paper_content}

问题: {query}

请仔细分析论文内容，准确回答问题。如果论文中没有相关信息，请说明。使用Markdown格式输出答案。"""
                else:
                    prompt = f"""Answer the question based on the paper content.

Paper:

{paper_content}

Question: {query}"""

                stream = await reasoning_client.chat.completions.create(
                    model=os.getenv("SCI_LLM_REASONING_MODEL", "deepseek-reasoner"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.2,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        
                        reasoning_content = getattr(delta, 'reasoning_content', None)
                        if reasoning_content:
                            print(f"[paper_qa] Reasoning: {reasoning_content}", flush=True)
                        
                        delta_content = delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"[paper_qa] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/ideation")
async def ideation(request: Request):
    """
    Ideation endpoint - uses embedding model for similarity and LLM for generation
    
    Request body:
    {
        "query": "Generate research ideas about climate change"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        # 硬编码的参考想法用于测试嵌入模型
        reference_ideas = [
            "Using deep learning to predict protein folding structures",
            "Applying transformer models to drug discovery and molecular design",
            "Leveraging reinforcement learning for automated experiment design",
            "Developing AI-powered literature review and knowledge synthesis tools",
            "Creating neural networks for climate modeling and weather prediction",
            "Using machine learning to analyze large-scale genomic datasets"
        ]

        print(f"[ideation] Received query: {query}")
        print(f"[ideation] Using {len(reference_ideas)} hardcoded reference ideas for embedding similarity")
        print(f"[ideation] Using LLM model: {os.getenv('SCI_LLM_MODEL')}")
        print(f"[ideation] Using embedding model: {os.getenv('SCI_EMBEDDING_MODEL')}")

        language = detect_language(query)
        print(f"[ideation] Detected language: {language}")

        async def generate():
            try:
                if language == 'zh':
                    prompt = f"""为以下研究领域生成创新的研究想法：

{query}"""
                else:
                    prompt = f"""Generate innovative research ideas for:

{query}"""

                print("[ideation] Computing embeddings for similarity analysis...")
                
                query_embedding = await get_embedding(query)
                
                if not query_embedding:
                    print("[ideation] Failed to get query embedding, generating ideas without similarity analysis")
                else:
                    similarities = []
                    for idx, idea in enumerate(reference_ideas):
                        idea_embedding = await get_embedding(idea)
                        if idea_embedding:
                            similarity = cosine_similarity(query_embedding, idea_embedding)
                            similarities.append((idx, idea, similarity))
                    
                    similarities.sort(key=lambda x: x[2], reverse=True)
                    
                    if language == 'zh':
                        prompt += f"\n\n参考想法（按相似度排序）：\n"
                    else:
                        prompt += f"\n\nReference ideas (ranked by similarity):\n"
                    
                    for idx, idea, sim in similarities:
                        prompt += f"\n{idx+1}. (similarity: {sim:.3f}) {idea}"
                    
                    if language == 'zh':
                        prompt += "\n\n基于以上参考想法生成新颖的研究想法。"
                    else:
                        prompt += "\n\nGenerate novel research ideas based on the above."

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"[ideation] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses LLM model with PDF content
    
    Request body:
    {
        "query": "Please review this paper",  # optional, default review prompt will be used
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "Please provide a comprehensive review of this paper")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_review] Received query: {query}")
        print(f"[paper_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        language = detect_language(query)
        print(f"[paper_review] Detected language: {language}")

        async def generate():
            try:
                pdf_text = extract_text_from_pdf_base64(pdf_content)
                if not pdf_text:
                    error_msg = "无法从PDF中提取文本" if language == 'zh' else "Failed to extract text from PDF"
                    error_data = {
                        "object": "error",
                        "message": error_msg
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                if language == 'zh':
                    prompt = f"""评审以下论文：

论文内容:

{pdf_text}

指令: {query}"""
                else:
                    prompt = f"""Review the following paper:

Paper:

{pdf_text}

Instruction: {query}"""

                stream = await client.chat.completions.create(
                    model=os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3,
                    stream=True
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            response_data = {
                                "object": "chat.completion.chunk",
                                "choices": [{
                                    "delta": {
                                        "content": delta_content
                                    }
                                }]
                            }
                            yield f"data: {json.dumps(response_data)}\n\n"

                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"[paper_review] 错误: {e}")
                error_data = {
                    "object": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
