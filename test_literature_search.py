#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文献搜索功能测试脚本
"""
import asyncio
from literature_search import search_literature, LiteratureSearcher


async def main():
    """测试文献搜索功能"""
    # 测试主题
    query = "transformer models"
    print(f"正在搜索主题: {query}\n")
    print("=" * 80)
    
    # 搜索文献（不提取PDF内容，速度更快）
    results = await search_literature(
        query=query,
        max_results=5,
        extract_pdf=False  # 设置为True会下载PDF并提取文本，速度较慢
    )
    
    print(f"\n找到 {len(results)} 篇文献:\n")
    print("=" * 80)
    
    for i, paper in enumerate(results, 1):
        print(f"\n【文献 {i}】")
        print(f"标题: {paper['title']}")
        print(f"来源: {paper['source']}")
        print(f"作者: {', '.join(paper['authors'][:5])}")
        if paper.get('published'):
            print(f"发表时间: {paper['published']}")
        print(f"摘要: {paper['abstract'][:200]}...")
        if paper.get('url'):
            print(f"链接: {paper['url']}")
        if paper.get('pdf_url'):
            print(f"PDF: {paper['pdf_url']}")
        print("-" * 80)
    
    # 如果需要提取PDF内容，可以这样测试（注意：会比较慢）
    print("\n\n是否要测试PDF内容提取？(这需要下载PDF文件，可能较慢)")
    print("取消注释下面的代码来测试PDF提取功能：")
    print("""
    # results_with_pdf = await search_literature(
    #     query=query,
    #     max_results=2,  # 只测试2篇，因为下载PDF较慢
    #     extract_pdf=True
    # )
    # 
    # for paper in results_with_pdf:
    #     if paper.get('pdf_text'):
    #         print(f"\\n论文: {paper['title']}")
    #         print(f"PDF文本长度: {len(paper['pdf_text'])} 字符")
    #         print(f"PDF文本预览: {paper['pdf_text'][:500]}...")
    """)


async def test_references():
    """测试获取文献及其引用信息"""
    print("=" * 80)
    print("测试：获取文献及其引用信息")
    print("=" * 80)
    
    searcher = LiteratureSearcher()
    try:
        # 搜索文献（使用OpenAlex和Crossref，因为它们支持引用信息）
        query = "transformer models"
        print(f"\n正在搜索主题: {query}\n")
        
        results = await searcher.search_and_extract(
            query=query,
            max_results=3,  # 只测试3篇文献
            extract_pdf=False,
            sources=['openalex', 'crossref'],  # 使用支持引用信息的源
            sort_by='citations'  # 按引用次数排序，更容易找到有引用信息的文献
        )
        
        print(f"\n找到 {len(results)} 篇文献\n")
        print("=" * 80)
        
        for i, paper in enumerate(results, 1):
            print(f"\n【文献 {i}】")
            print(f"标题: {paper.get('title', 'N/A')}")
            print(f"来源: {paper.get('source', 'N/A')}")
            if paper.get('authors'):
                print(f"作者: {', '.join(paper['authors'][:3])}")
            if paper.get('published'):
                print(f"发表时间: {paper['published']}")
            if paper.get('doi'):
                print(f"DOI: {paper['doi']}")
            if paper.get('url'):
                print(f"链接: {paper['url']}")
            
            # 显示引用文献数量
            referenced_works = paper.get('referenced_works', [])
            print(f"\n引用文献数量: {len(referenced_works)}")
            
            if referenced_works:
                print(f"引用文献列表（前10个）:")
                for j, ref_id in enumerate(referenced_works[:10], 1):
                    if isinstance(ref_id, str):
                        if ref_id.startswith('https://openalex.org/'):
                            print(f"  {j}. OpenAlex ID: {ref_id.split('/')[-1]}")
                        elif ref_id.startswith('http'):
                            print(f"  {j}. URL: {ref_id}")
                        else:
                            print(f"  {j}. DOI: {ref_id}")
            
            # 如果有DOI或OpenAlex ID，尝试获取详细的引用信息
            if paper.get('doi') or paper.get('openalex_id'):
                print(f"\n正在获取详细的引用文献信息...")
                try:
                    references = await searcher.get_references_for_paper(paper)
                    if references:
                        print(f"成功获取 {len(references)} 篇引用文献的详细信息:\n")
                        for k, ref in enumerate(references[:5], 1):  # 只显示前5个
                            print(f"  引用文献 {k}:")
                            print(f"    标题: {ref.get('title', 'N/A')}")
                            if ref.get('authors'):
                                print(f"    作者: {', '.join(ref['authors'][:3])}")
                            if ref.get('doi'):
                                print(f"    DOI: {ref['doi']}")
                            if ref.get('published'):
                                print(f"    发表时间: {ref['published']}")
                            if ref.get('url'):
                                print(f"    链接: {ref['url']}")
                            print()
                    else:
                        print("未找到详细的引用文献信息（可能是API限制或数据不存在）")
                except Exception as e:
                    print(f"获取引用信息时出错: {e}")
            else:
                print("\n该文献没有DOI或OpenAlex ID，无法获取详细引用信息")
            
            print("-" * 80)
    
    finally:
        await searcher.close()
    
    print("\n测试完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-references":
        # 运行引用信息测试
        asyncio.run(test_references())
    else:
        # 运行基本搜索测试
        asyncio.run(main())

