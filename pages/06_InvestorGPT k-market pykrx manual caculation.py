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
import datetime

# PyKRX 라이브러리 임포트
from pykrx import stock

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

# 재무제표 수집 및 재무 지표 계산 툴 정의
class FinancialStatementsArgsSchema(BaseModel):
    corp_code: str = Field(
        description="DART의 기업 고유 코드(corp_code). 예시: '00126380'"
    )
    stock_code: str = Field(
        description="기업의 종목 코드. 예시: '005930'"
    )

class FinancialStatementsTool(BaseTool):
    name = "FinancialStatementsTool"
    description = """
    기업의 최신 재무제표(연간 또는 분기)를 수집하고, PER, PBR, 부채비율 등을 계산합니다.
    """
    args_schema: Type[FinancialStatementsArgsSchema] = FinancialStatementsArgsSchema

    def _run(self, corp_code, stock_code):
        # 현재 날짜 기준으로 2년 전 날짜 계산 (데이터 부족 시 대비)
        today = datetime.datetime.now()
        two_years_ago = today - datetime.timedelta(days=730)
        bgn_de = two_years_ago.strftime('%Y%m%d')

        # 최신 보고서 목록 조회
        url = 'https://opendart.fss.or.kr/api/list.json'
        params = {
            'crtfc_key': dart_api_key,
            'corp_code': corp_code,
            'bgn_de': bgn_de,  # 시작일
            'pblntf_ty': 'A',  # 정기공시
            'page_count': '100'
        }
        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == '000' and data['list']:
            # 접수일자를 기준으로 내림차순 정렬하여 최신 보고서가 먼저 오도록 함
            reports = sorted(data['list'], key=lambda x: x['rcept_dt'], reverse=True)
            # 필요한 보고서 유형 (사업보고서, 반기보고서, 분기보고서)만 필터링
            report_types = ['사업보고서', '반기보고서', '분기보고서']
            for report in reports:
                report_nm = report.get('report_nm', '')
                if any(rtype in report_nm for rtype in report_types):
                    rcept_no = report['rcept_no']
                    bsns_year = report['rcept_dt'][:4]
                    # reprt_code 결정
                    if '사업보고서' in report_nm:
                        reprt_code = '11011'
                    elif '반기보고서' in report_nm:
                        reprt_code = '11012'
                    elif '분기보고서' in report_nm:
                        reprt_code = '11013'
                    else:
                        continue
                    break
            else:
                return "최신 재무제표를 찾을 수 없습니다."
        else:
            return f"재무제표 목록을 가져오는 중 오류 발생: {data.get('message', '알 수 없는 오류')}"

        # 재무제표 조회
        url = 'https://opendart.fss.or.kr/api/fnlttSinglAcnt.json'
        params = {
            'crtfc_key': dart_api_key,
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
            'fs_div': 'CFS'  # 연결재무제표
        }
        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == '000' and 'list' in data:
            fin_data = data['list']
            # 필요한 항목 추출
            fin_dict = {}
            for item in fin_data:
                account_nm = item['account_nm']
                if account_nm in ['당기순이익', '자본총계', '부채총계', '자산총계']:
                    amount = item['thstrm_amount']
                    if amount and amount != '':
                        fin_dict[account_nm] = int(amount.replace(',', ''))
            # 필요한 데이터가 있는지 확인
            required_fields = ['당기순이익', '자본총계', '부채총계', '자산총계']
            available_fields = [field for field in required_fields if field in fin_dict]

            if not available_fields:
                return "재무제표 데이터에 필요한 항목이 없습니다."

            # 발행주식수 조회
            shares_outstanding = self.get_shares_outstanding(corp_code)
            if shares_outstanding is None:
                shares_outstanding_available = False
            else:
                shares_outstanding_available = True

            # 주가 조회
            price = self.get_current_stock_price(stock_code)
            if price is None:
                return "주가를 가져오는 중 오류 발생"

            # 재무 지표 계산
            result = {}
            if '부채총계' in fin_dict and '자본총계' in fin_dict:
                debt_ratio = (fin_dict['부채총계'] / fin_dict['자본총계']) * 100
                result['부채비율'] = debt_ratio
            if '당기순이익' in fin_dict and '자본총계' in fin_dict:
                roe = (fin_dict['당기순이익'] / fin_dict['자본총계']) * 100
                result['ROE'] = roe
            if '당기순이익' in fin_dict and '자산총계' in fin_dict:
                roa = (fin_dict['당기순이익'] / fin_dict['자산총계']) * 100
                result['ROA'] = roa

            if shares_outstanding_available and '당기순이익' in fin_dict:
                eps = fin_dict['당기순이익'] / shares_outstanding
            else:
                eps = None

            if shares_outstanding_available and '자본총계' in fin_dict:
                bps = fin_dict['자본총계'] / shares_outstanding
            else:
                bps = None

            if eps is not None and eps != 0:
                per = price / eps
                result['PER'] = per
            else:
                result['PER'] = '데이터 없음'

            if bps is not None and bps != 0:
                pbr = price / bps
                result['PBR'] = pbr
            else:
                result['PBR'] = '데이터 없음'

            # 기타 정보 추가
            result.update({
                '당기순이익': fin_dict.get('당기순이익', '데이터 없음'),
                '자본총계': fin_dict.get('자본총계', '데이터 없음'),
                '부채총계': fin_dict.get('부채총계', '데이터 없음'),
                '자산총계': fin_dict.get('자산총계', '데이터 없음'),
                '발행주식수': shares_outstanding if shares_outstanding_available else '데이터 없음',
                '주가': price,
                '보고서명': report_nm,
                '보고서일자': report['rcept_dt']
            })
            return result

        else:
            return f"재무제표 데이터를 가져오는 중 오류 발생: {data.get('message', '알 수 없는 오류')}"

    def get_shares_outstanding(self, corp_code):
        # 발행주식수 조회
        url = 'https://opendart.fss.or.kr/api/stockTotqySttus.json'
        params = {
            'crtfc_key': dart_api_key,
            'corp_code': corp_code,
            'bsns_year': datetime.datetime.now().year,
            'reprt_code': '11011'  # 사업보고서
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == '000' and 'list' in data:
            total_shares = data['list'][0].get('distb_stock_co', None)
            if total_shares:
                return int(total_shares.replace(',', ''))
            else:
                return None
        else:
            return None

    def get_current_stock_price(self, stock_code):
        # 주가 조회
        try:
            today = datetime.datetime.now().strftime('%Y%m%d')
            df = stock.get_market_ohlcv_by_date(today, today, stock_code)
            if not df.empty:
                price = df['종가'].iloc[0]
                return price
            else:
                # 마지막 거래일 데이터 가져오기
                last_trading_day = stock.get_nearest_business_day_in_a_week(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')
                df = stock.get_market_ohlcv_by_date(last_trading_day, last_trading_day, stock_code)
                if not df.empty:
                    price = df['종가'].iloc[0]
                    return price
                else:
                    return None
        except:
            return None

# 주가 데이터 수집 툴 정의 (PyKRX 사용)
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
            # 오늘 날짜와 1년 전 날짜 계산
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            # 주가 데이터 가져오기
            df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code)
            if not df.empty:
                df.reset_index(inplace=True)
                df['날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
                # 열 이름을 영어로 변경
                df.rename(columns={
                    '날짜': 'Date',
                    '시가': 'Open',
                    '고가': 'High',
                    '저가': 'Low',
                    '종가': 'Close',
                    '거래량': 'Volume'
                }, inplace=True)
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

            기업의 최신 재무제표, PER, PBR, 부채비율 등의 재무 지표, 최신 주가 변동, 최신 뉴스, 재무 건전성 등을 종합적으로 분석하여 투자 의견과 그 이유를 제공합니다.

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