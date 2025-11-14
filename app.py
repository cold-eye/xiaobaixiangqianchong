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
- 研究背景：该研究领域的历史背景和重要性
- 关键主题：主要研究主题和核心概念
- 研究趋势：最新发展和新兴技术
- 研究空白：局限性、未解决的问题和未来研究方向

{citation_instruction}

请确保综述全面、准确、有深度，并在适当的地方使用作者年份格式的文献引用，如 (Smith, 2023)、(Smith et al., 2023)、Smith (2023) 或 Smith et al. (2023)。"""
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
- Background: Historical context and importance of the research area
- Key Themes: Main research themes and core concepts
- Current Trends: Recent developments and emerging technologies
- Research Gaps: Limitations, unsolved problems, and future directions

{citation_instruction}

Ensure the review is thorough, accurate, and insightful, and use author-year format citations like (Smith, 2023), (Smith et al., 2023), Smith (2023), or Smith et al. (2023) at appropriate places."""

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
                            
                            patterns = []
                            if year:
                                if multiple_authors:
                                    patterns.append(f"({re.escape(author_surname)} et al\\.?,? {year})")
                                    patterns.append(f"{re.escape(author_surname)} et al\\.? \\({year}\\)")
                                else:
                                    patterns.append(f"({re.escape(author_surname)},? {year})")
                                    patterns.append(f"{re.escape(author_surname)} \\({year}\\)")
                            else:
                                if multiple_authors:
                                    patterns.append(f"{re.escape(author_surname)} et al\\.")
                                else:
                                    patterns.append(re.escape(author_surname))
                            
                            cited = False
                            for pattern in patterns:
                                if re.search(pattern, full_text, re.IGNORECASE):
                                    cited = True
                                    print(f"[literature_review] 找到引用: {author_surname} ({year})")
                                    break
                            
                            if cited:
                                cited_refs.append(ref)
                    else:
                        print(f"[literature_review] 警告: full_text 为空，无法检测引用")
                        cited_refs = papers_references
                    
                    print(f"[literature_review] 检测到 {len(cited_refs)} 篇被引用的文献")
                    
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
                            
                            if ref['authors']:
                                if len(ref['authors']) == 1:
                                    authors_str = ref['authors'][0]
                                elif len(ref['authors']) == 2:
                                    authors_str = f"{ref['authors'][0]} & {ref['authors'][1]}"
                                else:
                                    authors_str = f"{ref['authors'][0]} et al."
                            else:
                                authors_str = 'Unknown Authors'
                            
                            if language == 'zh':
                                if year:
                                    ref_line = f"{authors_str} ({year}). {ref['title']}"
                                else:
                                    ref_line = f"{authors_str}. {ref['title']}"
                                if ref['published']:
                                    ref_line += f". {ref['published']}"
                                if ref['doi']:
                                    ref_line += f". DOI: {ref['doi']}"
                                elif ref['url']:
                                    ref_line += f". URL: {ref['url']}"
                            else:
                                if year:
                                    ref_line = f"{authors_str} ({year}). {ref['title']}"
                                else:
                                    ref_line = f"{authors_str}. {ref['title']}"
                                if ref['published']:
                                    ref_line += f". {ref['published']}"
                                if ref['doi']:
                                    ref_line += f". DOI: {ref['doi']}"
                                elif ref['url']:
                                    ref_line += f". URL: {ref['url']}"
                            
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
                
                if len(pdf_text) > 15000:
                    truncate_msg = "\n\n[文本已截断...]" if language == 'zh' else "\n\n[Text truncated...]"
                    pdf_text = pdf_text[:15000] + truncate_msg
                
                if language == 'zh':
                    prompt = f"""评审以下论文：

论文内容:

{pdf_text[:15000]}

指令: {query}"""
                else:
                    prompt = f"""Review the following paper:

Paper:

{pdf_text[:15000]}

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
