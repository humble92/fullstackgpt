def decompose_elements(soup, el, class_=None):
    if class_:
        foundings = soup.find_all(el, class_=class_)
    else:
        foundings = soup.find_all(el)

    # print(f"Number of {el} elements found: {len(foundings)}")  # 메뉴 개수 출력
    for element in foundings:
        # element.name이 None이면 태그 이름이 없는 것으로 간주되며, 이 경우 element를 문자열로 변환할 때 문제가 발생할 수 있습니다.
        # 하지만, 이 상황이 발생하는 이유는 보통 BeautifulSoup 객체가 잘못된 HTML을 파싱할 때입니다. 따라서, element와 element.name을 확인하는 것은 안전한 코딩 습관입니다.
        if element and element.name:
            # print(f"Before decompose: {element}")  # decompose 호출 전 내용 출력
            element.decompose()

    return soup
