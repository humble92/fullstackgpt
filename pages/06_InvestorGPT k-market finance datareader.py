# 필요한 라이브러리 임포트
import FinanceDataReader as fdr
from langchain.schema import SystemMessage
import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType

import pandas as pd
import xml.etree.ElementTree as ET
import datetime

# PyKRX 라이브러리 임포트
from pykrx import stock

# API 키 설정
dart_api_key = os.environ.get("DART_API_KEY")
kakao_api_key = os.environ.get("KAKAO_API_KEY")

# LLM 초기화
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

# 기업 코드 검색 툴 정의 (FinanceDataReader 사용)
class CompanyCodeSearchToolArgsSchema(BaseModel):
    company_name: str = Field(
        description="한글로 된 기업의 전체 이름. 예시: '삼성전자'"
    )

class CompanyCodeSearchTool(BaseTool):
    name = "CompanyCodeSearchTool"
    description = """
    입력된 기업명으로 종목 코드(stock_code)를 찾습니다.
    """
    args_schema: Type[CompanyCodeSearchToolArgsSchema] = CompanyCodeSearchToolArgsSchema

    def _run(self, company_name):
        try:
            df = fdr.StockListing('KRX')
            company_info = df[df['Name'] == company_name]
            if not company_info.empty:
                stock_code = company_info.iloc[0]['Code']
                return {'stock_code': stock_code}
            else:
                return f"회사명을 '{company_name}'으로 찾을 수 없습니다."
        except Exception as e:
            return f"기업 코드를 가져오는 중 오류 발생: {e}"

# 재무 지표 수집 툴 정의 (FinanceDataReader 사용)
class FinancialRatiosArgsSchema(BaseModel):
    stock_code: str = Field(
        description="기업의 종목 코드. 예시: '005930'"
    )

class FinancialRatiosTool(BaseTool):
    name = "FinancialRatiosTool"
    description = """
    기업의 주요 재무 지표(PER, PBR, ROE 등)를 수집합니다.
    """
    args_schema: Type[FinancialRatiosArgsSchema] = FinancialRatiosArgsSchema

    def _run(self, stock_code):
        try:
            df = fdr.DataReader(stock_code, exchange='KRX')
            if not df.empty:
                # 최신 데이터 가져오기
                latest_data = df.iloc[-1]
                per = latest_data.get('PER')
                pbr = latest_data.get('PBR')
                eps = latest_data.get('EPS')
                bps = latest_data.get('BPS')
                dividend_yield = latest_data.get('DIV')

                result = {
                    'PER': per,
                    'PBR': pbr,
                    'EPS': eps,
                    'BPS': bps,
                    '배당수익률': dividend_yield
                }
                return result
            else:
                return "재무 지표를 가져오는 중 오류 발생"
        except Exception as e:
            return f"재무 지표를 가져오는 중 오류 발생: {e}"

# 주가 데이터 수집 툴 정의 (FinanceDataReader 사용)
class StockPriceArgsSchema(BaseModel):
    stock_code: str = Field(
        description="기업의 종목 코드. 예시: '005930'"
    )

class StockPriceTool(BaseTool):
    name = "StockPriceTool"
    description = """
    기업의 최근 주가 데이터를 수집합니다.
    """
    args_schema: Type[StockPriceArgsSchema] = StockPriceArgsSchema

    def _run(self, stock_code):
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365)
            df = fdr.DataReader(stock_code, start_date, end_date)
            if not df.empty:
                df.reset_index(inplace=True)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                return df.to_dict(orient='records')
            else:
                return "주가 데이터를 가져오는 중 오류 발생"
        except Exception as e:
            return f"주가 데이터를 가져오는 중 오류 발생: {e}"

# 뉴스 데이터 수집 툴 정의
class NewsArgsSchema(BaseModel):
    query: str = Field(
        description="검색할 기업명"
    )

class NewsTool(BaseTool):
    name = "NewsTool"
    description = """
    기업과 관련된 최신 뉴스 기사를 수집합니다.
    """
    args_schema: Type[NewsArgsSchema] = NewsArgsSchema

    def _run(self, query):
        headers = {
            "Authorization": f"KakaoAK {kakao_api_key}"
        }
        url = "https://dapi.kakao.com/v2/search/web"
        params = {
            "query": query,
            "size": 10
        }
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if 'documents' in data:
            # 제목과 URL만 추출하여 반환
            news_list = []
            for doc in data['documents']:
                news = {
                    'title': doc['title'],
                    'url': doc['url']
                }
                news_list.append(news)
            return news_list
        else:
            return "뉴스 데이터를 가져오는 중 오류 발생"

# 에이전트 초기화
agent = initialize_agent(
    llm=llm,
    tools=[
        CompanyCodeSearchTool(),
        FinancialRatiosTool(),
        StockPriceTool(),
        NewsTool(),
    ],
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            당신은 전문적인 투자 분석가입니다.

            기업의 주요 재무 지표(PER, PBR, ROE 등), 최신 주가 변동, 최신 뉴스, 재무 건전성 등을 종합적으로 분석하여 투자 의견과 그 이유를 제공합니다.

            주어진 데이터를 바탕으로 기업의 현재 상황을 평가하고, 투자 추천 또는 비추천을 명확히 제시하세요.
            """
        )
    },
    handle_parsing_errors=True,
)

# Streamlit 애플리케이션 구성
st.set_page_config(
    page_title="InvestorGPT-Korea",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT-Korea

    기업명을 입력하시면 종합적인 투자 분석을 제공합니다.
    """
)

company = st.text_input("관심 있는 기업의 이름을 입력하세요.")

if company:
    with st.spinner('분석 중입니다...'):
        result = agent.invoke(company)
        st.write(result["output"])