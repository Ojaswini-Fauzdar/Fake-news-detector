import os
import tempfile
from typing import List
from pydantic import BaseModel, Field

import requests
from bs4 import BeautifulSoup
import concurrent.futures
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",          
    temperature=0.1,
    max_tokens=2048
)


class Claims(BaseModel):
    main_claims: List[str] = Field(..., description="List of 3-6 most important verifiable claims")

class Verdict(BaseModel):
    verdict: str = Field(..., description="REAL, FAKE, or SUSPICIOUS")
    confidence: float = Field(..., ge=0, le=1)
    explanation: str = Field(..., description="Detailed reasoning with key evidence")
    supporting_sources: List[str] = Field(default_factory=list)

claim_parser = PydanticOutputParser(pydantic_object=Claims)
verdict_parser = PydanticOutputParser(pydantic_object=Verdict)


claim_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact extraction assistant.
Extract only the most important, verifiable factual claims.
- For short text: return the main claim.
- For long articles: extract 3-6 key claims.
Be precise and concise."""),
    ("human", "Text:\n{text}\n\n{format_instructions}")
])

verdict_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional, unbiased fact-checker.
Use the provided context to evaluate the claims.
Be strict but fair. Always justify your verdict with specific evidence from the context."""),
    ("human", """Original Text:
{text}

Search Context:
{context}

{format_instructions}""")
])


def clean_text(text: str) -> str:
    for line in text.splitlines():
        lines = (line.strip() )
    result = []
    for line in lines:
        for phrase in line.split("  "):
            chunk = phrase.strip()
            if chunk:
                result.append(chunk)
    
    return '\n'.join(result)
    

def text_url(url: str) -> str | None:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, 'html.parser')
        
        
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "button", "form", "comments"]):
            tag.decompose()

        
        
        article = (
        soup.find('article') or
        soup.find(
           'div',
            class_=lambda x: x and (
               'article' in x.lower() or
               'story' in x.lower() or
               'content' in x.lower() or
               'post' in x.lower()
        )
    )
)
        if article:
            text = article.get_text(separator="\n") 
        else:
            text = soup.get_text(separator="\n")
        cleaned = clean_text(text)

        
        if len(cleaned) < 400:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                cleaned = docs[0].page_content if docs else cleaned
            except:
                pass

        return cleaned.strip()

    except Exception as e:
        print(f"URL Error {url}: {e}")
        return None


def text_pdf(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    
    try:
        loader = PyPDFLoader(path)
        pages = loader.load()
        return "\n\n".join(page.page_content for page in pages)
    finally:
        os.unlink(path)


def get_claims(text: str) -> List[str]:
    if not text or len(text.strip()) < 50:
        if text:
            return [text.strip()] 
        else:
            return []

    try:
        chain = claim_prompt | llm | claim_parser
        response = chain.invoke({
            "text": text[:14000],  
            "format_instructions": claim_parser.get_format_instructions()
        })
        claims = response.main_claims
        
        
        if not claims:
            return [text[:600]]
        
        return claims[:6]  
        
    except Exception as e:
        print("Claim extraction failed:", e)
        return [text[:700]]





def fetch_single_url(url: str, text_splitter) -> list:
    
    try:
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {"timeout": 6}
        loaded_docs = loader.load()
        split_docs = text_splitter.split_documents(loaded_docs)
        return split_docs[:2]
    except:
        return []

def search_claims(claims: List[str], k_per_claim=2): 
    serper = GoogleSerperAPIWrapper(k=5)  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    
    
    all_urls = []
    for claim in claims[:3]:  
        try:
            results = serper.results(claim)
            for item in results.get("organic", [])[:k_per_claim]:
                all_urls.append(item['link'])
        except Exception as e:
            print(f"Search error: {e}")
            continue

    
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for url in all_urls:
            futures = {executor.submit(fetch_single_url, url, text_splitter): url }
        for future in concurrent.futures.as_completed(futures, timeout=15):
            try:
                docs.extend(future.result())
            except:
                continue

    return docs[:12]  


def get_verdict(text: str, docs):
    try:
        
        context_parts = []
        for doc in docs[:6]:  
            content = doc.page_content[:1000]  
            if len(content.strip()) > 100:
                context_parts.append(content)
        if context_parts:
            context = "\n\n---\n\n".join(context_parts)  
        else:
            context = "No relevant web context found."

        chain = verdict_prompt | llm | verdict_parser
        result = chain.invoke({
            "text": text[:6000],   
            "context": context[:8000],  
            "format_instructions": verdict_parser.get_format_instructions()
        })

        if not result.supporting_sources and docs:
            for doc in docs[:5]:
                result.supporting_sources = [doc.metadata.get('source', 'Unknown') ]

        return result

    except Exception as e:
        print("Verdict Error:", str(e))
        return Verdict(
            verdict="SUSPICIOUS",
            confidence=0.45,
            explanation=f"Analysis error: {str(e)[:200]}",
            supporting_sources=[]
        )
