# 필요한 라이브러리 임포트
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
import yfinance as yf
import datetime

# API 키 설정
dart_api_key = os.environ.get("DART_API_KEY")
kakao_api_key = os.environ.get("KAKAO_API_KEY")

# LLM 초기화
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

# 기업 코드 검색 툴 정의
class CompanyCodeSearchToolArgsSchema(BaseModel):
    company_name: str = Field(
        description="한글로 된 기업의 전체 이름. 예시: '삼성전자'"
    )

class CompanyCodeSearchTool(BaseTool):
    name = "CompanyCodeSearchTool"
    description = """
    입력된 기업명으로 DART의 기업 고유 코드(corp_code)와 종목 코드(stock_code)를 찾습니다.
    """
    args_schema: Type[CompanyCodeSearchToolArgsSchema] = CompanyCodeSearchToolArgsSchema

    def __init__(self):
        super().__init__()
        self.corp_code_dict = self._load_corp_codes()

    def _load_corp_codes(self):
        # corpCode.xml 파일이 없으면 다운로드
        if not os.path.exists('corpCode.xml'):
            url = f'https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={dart_api_key}'
            response = requests.get(url)
            with open('corpCode.xml', 'wb') as f:
                f.write(response.content)
        # XML 파싱하여 기업명과 코드 매핑 생성
        tree = ET.parse('corpCode.xml')
        root = tree.getroot()
        corp_code_dict = {}
        for child in root.findall('list'):
            corp_code = child.find('corp_code').text.strip()
            corp_name = child.find('corp_name').text.strip()
            stock_code = child.find('stock_code').text.strip()
            corp_code_dict[corp_name] = {
                'corp_code': corp_code,
                'stock_code': stock_code
            }
        return corp_code_dict

    def _run(self, company_name):
        result = self.corp_code_dict.get(company_name)
        if result and result['stock_code']:
            return result
        else:
            return f"회사명을 '{company_name}'으로 찾을 수 없거나 상장되지 않은 기업입니다."

# 재무제표 수집 툴 정의 (최신 보고서 사용)
class FinancialStatementsArgsSchema(BaseModel):
    corp_code: str = Field(
        description="DART의 기업 고유 코드(corp_code). 예시: '00126380'"
    )

class FinancialStatementsTool(BaseTool):
    name = "FinancialStatementsTool"
    description = """
    기업의 최신 재무제표(연간 또는 분기)를 수집합니다.
    """
    args_schema: Type[FinancialStatementsArgsSchema] = FinancialStatementsArgsSchema

    def _run(self, corp_code):
        # 현재 날짜 기준으로 1년 전 날짜 계산
        today = datetime.datetime.now()
        one_year_ago = today - datetime.timedelta(days=365)
        bgn_de = one_year_ago.strftime('%Y%m%d')

        # 최신 보고서 접수번호 조회
        url = 'https://opendart.fss.or.kr/api/list.json'
        params = {
            'crtfc_key': dart_api_key,
            'corp_code': corp_code,
            'bgn_de': bgn_de,  # 시작일을 동적으로 설정
            'pblntf_detail_ty': 'A001',  # 정기공시
            'page_count': '100'
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == '000' and data['list']:
            # 사업보고서, 반기보고서, 분기보고서 순으로 검색
            report_types = ['사업보고서', '반기보고서', '분기보고서']
            rcept_no = None
            for report_type in report_types:
                for report in data['list']:
                    report_nm = report.get('report_nm', '')
                    if report_type in report_nm:
                        rcept_no = report['rcept_no']
                        bsns_year = report['rcept_dt'][:4]
                        # reprt_code 설정
                        if report_type == '사업보고서':
                            reprt_code = '11011'
                        elif report_type == '반기보고서':
                            reprt_code = '11012'
                        elif report_type == '분기보고서':
                            reprt_code = '11013'
                        else:
                            continue
                        break
                if rcept_no:
                    break
            if not rcept_no:
                return "최신 재무제표를 찾을 수 없습니다."
        else:
            return f"재무제표 목록을 가져오는 중 오류 발생: {data.get('message', '알 수 없는 오류')}"

        # 재무제표 조회
        url = 'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json'
        params = {
            'crtfc_key': dart_api_key,
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
            'fs_div': 'CFS'  # 연결재무제표
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == '000':
            return data['list']
        else:
            return f"재무제표 데이터를 가져오는 중 오류 발생: {data.get('message', '알 수 없는 오류')}"

# 주가 데이터 수집 툴 정의 (최신 데이터 사용)
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
            # 종목 코드에 따라 거래소 코드 추가
            ticker = f"{stock_code}.KS" if not stock_code.startswith('A') else f"{stock_code[1:]}.KS"  # 코스피 종목의 경우
            data = yf.Ticker(ticker)
            # 오늘 날짜와 1년 전 날짜 계산
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365)
            hist = data.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not hist.empty:
                # 날짜 데이터를 문자열로 변환하여 JSON 직렬화 가능하도록 처리
                hist.reset_index(inplace=True)
                hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
                return hist.to_dict(orient='records')
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
        # CompanyCodeSearchTool(),
        FinancialStatementsTool(),
        StockPriceTool(),
        NewsTool(),
    ],
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            당신은 전문적인 투자 분석가입니다.

            기업의 최신 재무제표, 최신 주가 변동, 최신 뉴스, 재무 건전성 등을 종합적으로 분석하여 투자 의견과 그 이유를 제공합니다.

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