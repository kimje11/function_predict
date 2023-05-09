import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import string
import math
from sklearn.metrics import mean_squared_error
from decimal import Decimal





def create_dict(number, title):
    # 변수의 수에 따라 알파벳 순서대로 키 생성
    # keys = list(string.ascii_uppercase)[:num_vars]
    keys=title
    print(title)
    # 각 변수에 대한 값을 빈 문자열("")로 초기화하는 딕셔너리 생성
    values = {key: "" for key in keys}
    values.update({"사용":True})

    return values

def create_dict_abc(num_vars):
    # 변수의 수에 따라 알파벳 순서대로 키 생성
    keys = list(string.ascii_uppercase)[:num_vars]
    # 각 변수에 대한 값을 빈 문자열("")로 초기화하는 딕셔너리 생성
    values = {key: "" for key in keys}
    values.update({"사용":True})
    return values

def dynamic_slider(min_val, max_val,name):
    # 최소/최대값 조정용 슬라이더
    # slider_range = st.slider(name, min_value=min_val, max_value=max_val)
    # print("슬라이더 표시",slider_range)
    # # 최소/최대값 조정 슬라이더
    # min_val = st.slider('Minimum Value', min_value=-5, max_value=max_val-5)
    # max_val = st.slider('Maximum Value', min_value=min_val, max_value=10)
    i=0
    if i==0:
        value=st.number_input(name,-10, 10, 0, step=1)
        i+=1
    else:
        value=st.number_input(name,-10, 10, value)
    # 업데이트된 최소/최대값으로 슬라이더 다시 생성
    slider_range = st.slider(name, min_value=value-5, max_value=value+5, value=value)
    return slider_range


# from fbprophet import Prophet

# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False

#______________________폰트 깨짐__________________________________


import matplotlib.font_manager as fm
import os

# 폰트 경로 설정
# 현재 작업 중인 디렉토리의 파일과 폴더를 나열합니다.
files_and_directories = os.listdir()
font_path = files_and_directoried[5]
print(font_path)
# FontProperties 인스턴스 생성
font_prop = fm.FontProperties(fname=font_path, size=10)
plt.rc('font', family=font_prop.get_name())



#초기 리스트 정리
title_count = st.session_state.get("title_count", 0)
title_list = st.session_state.get("title_list", [])
button_clicked = st.session_state.get("button_clicked", False)
show_checked = st.session_state.get("show_checked", False)


data = []


with st.sidebar:
    st.title("실생활 데이터를 활용한 예측 함수 만들기")
    select = st.selectbox(
        "",
        key="visibility",
        options=["데이터 선택","직접입력", "세계인구증가"],
    )
    if select == "직접입력":
        number = st.number_input('변인 갯수', value=2)
        title_edit=st.checkbox("변인 이름 수정")
        keys = list(string.ascii_uppercase)[:number]
        if title_count==0:
            if title_edit==False:
                for i in range(number):
                    data.append(create_dict_abc(number))
                    if len(title_list)<i+1:
                        title_list.append(keys[i])
                        
                st.session_state["title_list"] = title_list
                print("제목", title_list)                   
            else:
                print("여기서",title_list)
                for i in range(number):
                    globals()["title"+str(i)]=st.text_input(f"{i+1}번째 변인 이름", value=title_list[i])
                    title_list[i]=globals()["title"+str(i)]
                title_count += 1
                st.session_state["title_count"] =1
                st.session_state["title_list"] = title_list
        elif title_edit==True:
            print("여기서",title_list)
            new_ord = number-len(title_list)
            ex_ord = len(title_list)
            print("갯수:", new_ord)
            if new_ord >=0:
                for i in range(ex_ord):
                    globals()["title"+str(i)]=st.text_input(f"{i+1}번째 변인 이름", value=title_list[i])
                    title_list[i]=globals()["title"+str(i)]
                for i in range(new_ord):
                    keys = list(string.ascii_uppercase)[:number]
                    globals()["title"+str(i+ex_ord)]=st.text_input(f"{i+ex_ord+1}번째 변인 이름", value=keys[i+ex_ord])
                    title_list.append(globals()["title"+str(i+ex_ord)])
            elif new_ord<0:
                title_list.pop()
                # new_ord = number-len(title_list)
                ex_ord = len(title_list)
                for i in range(ex_ord):
                    globals()["title"+str(i)]=st.text_input(f"{i+1}번째 변인 이름", value=title_list[i])
                    title_list[i]=globals()["title"+str(i)]
                # for i in range(new_ord):
                #     keys = list(string.ascii_uppercase)[:number]
                #     globals()["title"+str(i+ex_ord)]=st.text_input(f"{i+ex_ord+1}번째 변인 이름", value=keys[i+ex_ord])
                #     title_list.append(globals()["title"+str(i+ex_ord)])
            # else:
            #     for i in range(ex_ord):
            #             globals()["title"+str(i)]=st.text_input(f"{i+1}번째 변인 이름", value=title_list[i])
            #             title_list[i]=globals()["title"+str(i)]
            #     # for i in range(new_ord):
            #     #     keys = list(string.ascii_uppercase)[:number]
            #     #     globals()["title"+str(i+ex_ord)]=st.text_input(f"{i+ex_ord+1}번째 변인 이름", value=keys[i+ex_ord])
            #     #     title_list.append(globals()["title"+str(i+ex_ord)])
            # print("여기서",title_list)
            st.session_state["title_list"] = title_list
            
    st.sidebar.markdown("---")
    st.write("예측 함수 선택")
    check1 = st.checkbox("일차함수로 예측해보기")
    check2 = st.checkbox("지수함수로 예측해보기")
    # check3 = st.checkbox("다항함수로 예측해보기")
    # check4 = st.checkbox("삼각함수로 예측해보기")
    # check5 = st.checkbox("인공지능모델로 예측해보기")
    st.sidebar.markdown("---")
    check6 = st.checkbox("나의 함수 그래프 감추기")
    if check1 and not check6:
        st.latex('y = ax + b')
        # a=dynamic_slider(0,10,'기울기')
        num_step=float(0.1)
        a = st.number_input('기울기(a)', step=0.1)
        b = st.number_input('y절편(b)',  step=0.1)
    elif check2 and not check6:
        st.latex('y = a\\cdot b^{x}')
        exp_a = st.number_input('지수함수의 계수(a)', step=0.1)
        exp_b = st.number_input('지수함수의 밑(b)', step=0.1)
        # exp_c = st.sidebar.slider('지수함수의 점근선(c)', 0.0, 10000.0, 0.0)
    

if select=="데이터 선택":
    st.title("평균제곱 오차(RMSE) 이해하기")
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2}")
    image_url = 'https://miro.medium.com/v2/resize:fit:640/format:webp/1*jopCO2kMEI84s6fiGKdXqg.png'
    st.image(image_url, use_column_width=True)
    
    st.write("""RMSE는 Root Mean Squared Error(평균 제곱근 오차)의 약어로, 머신 러닝 모델의 예측값과 실제 값 사이의 차이를 평가하는 지표입니다. 이를 계산하기 위해서는 예측값과 실제 값의 차이를 제곱한 후, 모든 차이의 평균을 구한 후, 다시 제곱근을 취해야 합니다.
이렇게 계산된 RSME 값은 오차의 절대적인 크기를 나타내며, 값이 작을수록 모델이 예측을 잘 수행한 것입니다. RSME는 머신 러닝 모델을 평가하는 지표로 널리 사용되며, 모델의 성능을 개선하는 데 도움이 됩니다.""")

else:
    title = st.text_input('무슨 데이터인가요?', '여기에 입력하세요.')
    if select=="직접입력":
        if title_count == 1:
            for i in range(3):
            # data.append({"x": "", "y": "", "사용": True})
                data.append(create_dict(number,title_list))
    elif select=="세계인구증가":
        data=[{"t(1900년이후의 햇수)": 0, "인구(백만)": 1650, "사용": True},
                        {"t(1900년이후의 햇수)": 10, "인구(백만)": 1750, "사용": True},
                        {"t(1900년이후의 햇수)": 20, "인구(백만)": 1860, "사용": True},
                        {"t(1900년이후의 햇수)": 30, "인구(백만)": 2070, "사용": True},
                        {"t(1900년이후의 햇수)": 40, "인구(백만)": 2300, "사용": True},
                        {"t(1900년이후의 햇수)": 50, "인구(백만)": 2560, "사용": True},
                        {"t(1900년이후의 햇수)": 60, "인구(백만)": 3040, "사용": True},
                        {"t(1900년이후의 햇수)": 70, "인구(백만)": 3710, "사용": True},
                        {"t(1900년이후의 햇수)": 80, "인구(백만)": 4450, "사용": True},
                        {"t(1900년이후의 햇수)": 90, "인구(백만)": 5280, "사용": True},
                        {"t(1900년이후의 햇수)": 100, "인구(백만)": 6080, "사용": True},
                        {"t(1900년이후의 햇수)": 110, "인구(백만)": 6870, "사용": True}]

    df = pd.DataFrame(data)
    edited_df = st.experimental_data_editor(df, use_container_width=True, num_rows="dynamic")
    # 체크된 행만 가져옵니다.
    edited_df = edited_df[edited_df["사용"] == True]


    #마지막줄 사용으로 바꾸기
    # edited_df["사용"] = True

    # 캔버스 사이즈 적용
    # plt.rcParams["figure.figsize"] = (12, 9)

    # 히스토그램 생성

    x_column = st.selectbox('X 축 열 선택', edited_df.columns[:-1])
    y_column = st.selectbox('Y 축 열 선택', edited_df.columns[:-1], index=1)
    try:
        x_min = edited_df[x_column].astype(float).min()
        x_max = edited_df[x_column].astype(float).max()
        y_min = edited_df[y_column].astype(float).min()
        y_max = edited_df[y_column].astype(float).max()
        x_range_max=int(x_max)//10+2
        y_range_max=int(y_max)//10+2



        fig, ax = plt.subplots()
        x = [x for x in range(int(x_min), int(x_max)+x_range_max)]
        exp_x = np.linspace(int(x_min), int(x_max), 100)
        if check1 and not check6:
            y_values = [a*x + b for x in edited_df[x_column].astype(float)]
            y = [a*x_i+b for x_i in x]
            
            rmse = np.sqrt(mean_squared_error(edited_df[y_column].astype(float), y_values))
            
            ax.plot(x, y)

        elif check2 and not check6:
            
            exp_y = exp_a * exp_b**exp_x
            ax.plot(exp_x,exp_y)
            

            real_y= edited_df[y_column].astype(float)
            test_y= np.array([exp_a*(exp_b**x) for x in edited_df[x_column].astype(float)])
            exp_rmse = np.sqrt(mean_squared_error(real_y, test_y))
            
            # equation = f'인공지능 예측함수: y = {z[0]:.2f}x + {z[1]:.2f}'
            # ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
            exp_equation = f"my_predict: y={exp_a}*{exp_b}^x, RMSE: {exp_rmse:.2f}"
            # exp_equation = f"나의 예측: y={a}x+{b}, RMSE: {exp_rmse:.2f}"
            # ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
            ax.annotate(exp_equation, xy=(0.05, 0.95), xycoords='axes fraction',  fontsize=12, va='top', ha='left', bbox=dict(boxstyle='round', fc='w'))
           
            # x와 y 데이터를 정의합니다.
            # x = [[0], [10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110]]
            # y = [1650,1750,1860,2070,2300,2560,3040,3710,4450,5280,6080,6870]

        elif check3:
            df['y'] = np.log(edited_df[y_column].astype(float))
            X = edited_df[y_column].astype(float)
            y = df['y']
        
        ax.set_ylim(0,y_max+y_range_max)
        ax.scatter(edited_df[x_column].astype(float), edited_df[y_column].astype(float), alpha=0.5)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        # if check1:
        #     ax.plot(edited_df[x_column].astype(float), p(edited_df[x_column].astype(float)), 'r--')
        #     equation = f'인공지능 예측함수: y = {z[0]:.2f}x + {z[1]:.2f}'
        #     ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
        # elif check2:
        #     # 지수 회귀 모델을 학습합니다.
        #     x = np.array(edited_df[x_column].astype(float))  # 1차원 배열
        #     x_2d = x.reshape(-1, 1)  # 2차원 배열로 변환 
        #     model = LinearRegression().fit(x_2d, np.log(edited_df[y_column].astype(float)))
        
        #     # x_range = np.arange(0,200, 10)
        #     y_pred = np.exp(model.intercept_) * np.exp(model.coef_ * exp_x)
        #     ax.plot(exp_x, y_pred, color='red')

        # print(edited_df[y_column])
        # # 축 레이블 설정
        # ax.set_xlabel(x_column)
        # ax.set_ylabel(y_column)

    

    

            
        if check1:
            # y = [a*x_i+b for x_i in x]
            z = np.polyfit(edited_df[x_column].astype(float), edited_df[y_column].astype(float), 1)
            poly = np.poly1d(z)
            # y_pred = np.polyval(z, x)
            y_pred=poly(edited_df[x_column].astype(float))
            print("예측값",y_pred)
            print(y_pred)
            # ai_rmse = np.sqrt(mean_squared_error(edited_df[y_column].astype(float), y_pred))
            decimal_y = [Decimal(str(x)) for x in edited_df[y_column]]
            decimal_y_pred=[round(Decimal(str(x)),2) for x in y_pred]
            ai_rmse = np.sqrt(mean_squared_error(y_pred, y_pred))
            print(decimal_y, decimal_y_pred)
            ai_rmse = np.sqrt(mean_squared_error(decimal_y, decimal_y_pred))
            # ax.plot(x, y)
            if not check6:
                equation = f"my_predict: y={a}x+{b}, RMSE: {rmse:.2f}"
                # ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
                ax.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',  fontsize=12, va='top', ha='left', bbox=dict(boxstyle='round', fc='w'))
                # equation2=(f"RMSE: {rmse:.2f}")
                # ax.annotate(equation2, xy=(0.05, 1.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
            check_function = st.checkbox("인공지능이 예측한 일차함수 확인해보기")
            

            if check_function:
                ax.plot(edited_df[x_column].astype(float), poly(edited_df[x_column].astype(float)), 'r--')
    
                # equation = f'인공지능 예측함수: y = {z[0]:.2f}x + {z[1]:.2f}'
                
                st.latex(f' y={z[0]:.2f}x+{z[1]:.2f},RMSE={ai_rmse}')
                # st.write(f"RMSE: {rmse:.2f}")
                number = st.number_input('x값을 입력하여 예측값을 확인해보세요.', step=1)
                st.write('y값은', z[0]*number+z[1])
                
                # 산점도 출력
            st.pyplot(fig)

        if check2:
            x2 = np.array(edited_df[x_column].astype(float))  # 1차원 배열
            y2 = np.array(edited_df[y_column].astype(float))
            y_log = np.log(y2)
            x_2d = x2.reshape(-1, 1)  # 2차원 배열로 변환 
            model = LinearRegression().fit(x_2d, y_log)
            # RMSE 계산
            y_log_pred = model.predict(x_2d)
            pred_y = np.exp(y_log_pred)
            print("predy",pred_y)
            rmse = np.sqrt(mean_squared_error(y2, pred_y))
            print(f"RMSE: {rmse:.2f}")
            # x_range = np.arange(0,200, 10)
            y_pred = np.exp(model.intercept_) * np.exp(model.coef_ * exp_x)
            
            # 모델의 방정식을 출력합니다.
            # st.write('나의 예측: y = %.2f e^(%.2f x)' % (exp_a, exp_b))
            check_function = st.checkbox("인공지능이 예측한 지수함수 확인해보기")
        
            if check_function:
                ax.plot(exp_x, y_pred, color='red')
                st.latex('y = %.2f e^{%.2f x}, RMSE= %.2f' % (np.exp(model.intercept_), model.coef_[0],rmse))
                e_power = math.exp(model.coef_[0])
                e_power_rounded = round(e_power, 4)
                st.latex('[참고]  e^{%.2f} 의 근삿값은 %.2f 입니다.' % (model.coef_[0], e_power_rounded))
                number = st.number_input('x값을 입력하여 예측값을 확인해보세요.', step=1)
                st.write('y값은', np.exp(model.intercept_)*np.exp(model.coef_[0]*number))
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                # 산점도 출력
            st.pyplot(fig)
    #Propeht으로 예측

    except Exception as e:
        print(e)
        st.warning("자료를 입력하세요.")



        # 산점도 출력
        # st.pyplot(fig)
        
        # favorite_command = edited_df.loc[edited_df["y"].idxmax()]["x"]
        # st.markdown(f"Your favorite command is **{favorite_command}** 🎈")
        # option = st.selectbox(
        #     "How would you like to be contacted?",
        #     ("Email", "Home phone", "Mobile phone"),
        #     label_visibility=st.session_state.visibility,
        #     disabled=st.session_state.disabled,
        # )
    # Sidebar 메뉴 생성

    # option = st.selectbox(
    #     '자료 선택',
    #     ('직접입력', '세계인구변화', 'Mobile phone'))
    # st.write('You selected:', option)


    # menu = ['메뉴1', '메뉴2', '메뉴3']
    # choice = st.sidebar.radio('메뉴', menu)

    # # 각 메뉴에 맞는 동작 정의
    # if choice == '메뉴1':
    #     # 메뉴1 선택 시 수행할 동작 정의
    #     st.write('메뉴1 선택됨')
    # elif choice == '메뉴2':
    #     # 메뉴2 선택 시 수행할 동작 정의
    #     st.write('메뉴2 선택됨')
    # else:
    #     # 메뉴3 선택 시 수행할 동작 정의
    #     st.write('메뉴3 선택됨')


    # df = pd.DataFrame({
    #   'first column': [1, 2, 3, 4],
    #   'second column': [10, 20, 30, 40]
    # })

    # df

# 버튼의 스타일을 변경합니다.
button_style = """
<style>
.stButton button {
    background-color: #007bff;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 5px;
}
</style> """
