import streamlit as st
import pandas as pd
import pickle
import csv
from gensim import corpora, models, similarities
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import re
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
from pyspark.sql.functions import broadcast
from st_aggrid import AgGrid
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("Streamlit ALS").getOrCreate()

spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()


st.title("Data Science Project")
st.write("## HASAKI_RECOMMENDER_SYSTEM")


st.sidebar.title("Menu")
menu = st.sidebar.radio("Select a recommendation method:", ["Content-Based","Collaborative Filtering"])

st.sidebar.write("""#### Thành viên thực hiện:
                 Trần Đình Hùng
    Phạm Thị Ngọc Huyền """)
st.sidebar.write("""#### Giảng viên hướng dẫn:
    Khuất Thùy Phương """)
st.sidebar.write("""#### Thời gian báo cáo: 12/2024""")

# Đọc dữ liệu sản phẩm
df_products = pd.read_csv('San_pham.csv',encoding='utf-8')
df_products['gia_goc'] = df_products['gia_goc'].fillna(df_products['gia_ban']) # giá gốc trống NaN thì thay bằng giá bán

#Phần 1" Content_based
if menu=="Content-Based":
    st.header("Content-Based")

    df_products_filtered = df_products[df_products['diem_trung_binh'] >= 4.5]
    df_products_sorted = df_products_filtered.sort_values(by='ten_san_pham', ascending=True)

    #upload file pkl
    def load_pipeline():
        with open('cosine_pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline

    pipeline = load_pipeline()
    vectorizer = pipeline['vectorizer']
    tfidf_matrix = pipeline['tfidf_matrix']
    cosine_sim = pipeline['cosine_sim']
    df = pipeline['df']

    # function cần, chọn sản phẩm trong list
    def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
        # Get the index of the product that matches the ma_san_pham
        matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
        if not matching_indices:
            print(f"No product found with ID: {ma_san_pham}")
            return pd.DataFrame()  # Return an empty DataFrame if no match
        idx = matching_indices[0]

        # Get the pairwise similarity scores of all products with that product
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the products based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the nums most similar products (Ignoring the product itself)
        sim_scores = sim_scores[1:nums+1]

        # Get the product indices
        product_indices = [i[0] for i in sim_scores]

        # Filter the DataFrame to include only highly-rated products
        filtered_df = df.iloc[product_indices]
        filtered_df = filtered_df[filtered_df['diem_trung_binh'] >= 4.5]

        # Return the top n most similar products as a DataFrame
        return filtered_df.head(nums)

    # function gõ từ khóa tìm sản phẩm
    def get_recommendations_by_content(user_input, vectorizer,tfidf_matrix, df, min_rating=4.5, nums=5):
        # Transform user input into a vector
        user_vector = vectorizer.transform([user_input])

        # Compute cosine similarity between user input and all product vectors
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Add similarity scores to the copied DataFrame
        df_copy['similarity_score'] = similarities

        # Step 1: Filter out products with ratings below the minimum threshold
        high_rating_products = df_copy[df_copy['diem_trung_binh'] >= min_rating]

        # Step 2: Sort by similarity score in descending order
        high_rating_sorted = high_rating_products.sort_values(by='similarity_score', ascending=False)

        # Step 3: Check if there are enough high-rated products with high similarity scores
        if len(high_rating_sorted) >= nums:
            # Return the top products by similarity score
            return high_rating_sorted[['ten_san_pham', 'ma_san_pham', 'diem_trung_binh', 'gia_ban', 'gia_goc']].head(nums)
        else:
            # Handle case where not enough high-rated products meet criteria
            return high_rating_sorted[['ten_san_pham','ma_san_pham', 'diem_trung_binh', 'gia_ban','gia_goc']]

    # function Hiển thị đề xuất theo sản phẩmm trong dropbox ra bảng
    def display_recommended_products(recommended_products, cols=5):
        for i in range(0, len(recommended_products), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(recommended_products):
                    product = recommended_products.iloc[i + j]
                    
                    with col:   
                        st.write(product['ten_san_pham'])
                        st.write(f"**Giá bán:** {product['gia_ban']:,} VNĐ")  # Format price with commas
                        st.write(f"**Đánh giá:** {product['diem_trung_binh']}/5")

                        expander = st.expander(f"Mô tả")
                        product_description = product['mo_ta']
                        truncated_description = ' '.join(product_description.split()[:90]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None

    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in df_products_sorted.iterrows()]

    # tạo các tab
    tab1, tab2= st.tabs(["Chọn sản phẩm", "Gõ từ khóa liên quan đến SP"])
    with tab1:
        ###### Giao diện Streamlit ######
        st.image('hasaki_banner.jpg', use_container_width=True)
        # Tạo một dropdown với options là các tuple này
        selected_product = st.selectbox(
            "Chọn sản phẩm",
            options=product_options,
            format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
            )
        # Display the selected product
        st.write("Bạn đã chọn:", selected_product)

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_ma_san_pham = selected_product[1]

        if st.session_state.selected_ma_san_pham:
            st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
            # Hiển thị thông tin sản phẩm được chọn
            selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

            if not selected_product.empty:
                st.write('#### Bạn vừa chọn:')
                st.write('### ', selected_product['ten_san_pham'].values[0])

                product_description = selected_product['mo_ta'].values[0]
                truncated_description = ' '.join(product_description.split()[:90])
                st.write('##### Thông tin:')
                st.write(truncated_description, '...')

                # Add Selling Price and Rating
                selling_price = selected_product['gia_ban'].values[0]  # Get selling price
                rating = selected_product['diem_trung_binh'].values[0]  # Get rating

                st.write('**Giá bán:**', f"{selling_price:,} VNĐ")  # Format price with commas
                st.write('**Đánh giá:**', f"{rating}/5")  # Display rating
                st.markdown("### ----------------------------------------------")
                st.write('##### Các sản phẩm liên quan:')
                recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, cosine_sim, nums=5) 
                display_recommended_products(recommendations, cols=5)
                st.image('top_10_best_sellers_updated.png', use_container_width=True)
            else:
                st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
    with tab2:
        ###### Giao diện Streamlit ######
        st.image('HASAKI_5.jpg', use_container_width=True)
        user_input = st.text_input("Nhập từ khóa sản phẩm cần tìm:")
        min_rating = 4.5

        if user_input:
            recommendations = get_recommendations_by_content(user_input, vectorizer, tfidf_matrix, df, min_rating)
            if recommendations.empty:
                st.write("Không tìm thấy sản phẩm tương ứng. Vui lòng nhập lại")
            else:
                st.write("Các sản phẩm bạn có thể tham khảo:")
                st.dataframe(recommendations)
                st.image('top_10_best_sellers_updated.png', use_container_width=True)
#phần 2 Collaborative
else:
    
    st.header("Collaborative Filtering")
    ###### Giao diện Streamlit ######
    st.image('HASAKI_4.jpg', use_container_width=True)
    #đọc file
    
    df=pd.read_csv('Processing_Danh_gia_HASAKI_VS5.csv')
    df_customer=pd.read_csv('Khach_hang.csv')

    from pyspark.sql.functions import broadcast
    df_products_spark = spark.createDataFrame(df_products)
    df_products_spark.write.parquet("optimized_df_products.parquet", mode="overwrite")

    model = ALSModel.load("model.file")
    df_asin_asin_idx = spark.read.parquet("optimized_df_asin.parquet")
    new_user_recs = spark.read.parquet("optimized_new_user_recs.parquet")
    df_products_spark = spark.read.parquet("optimized_df_products.parquet")
    
    # Merge df and df_products to calculate customer spending
    merged_df = pd.merge(df, df_products, on="ma_san_pham", how="inner")
    

    # Classify customers based on spending
    def classify_customer(total_spending):
        if total_spending >= 7000000:
            return "Thành viên Kim Cương"
        elif total_spending >= 5000000:
            return "Thành viên Vàng"
        elif total_spending >= 3000000:
            return "Thành viên Bạc"
        else:
            return "Khách hàng thân thiết"

    ## Calculate total spending and classify customers & ma_khach_hang is a string 
    customer_spending = merged_df.groupby('ma_khach_hang')['gia_ban'].sum().reset_index()
    customer_spending['membership'] = customer_spending['gia_ban'].apply(classify_customer)
    
    df_customer["ma_khach_hang"] = df_customer["ma_khach_hang"].astype(str)
    customer_spending['ma_khach_hang'] = customer_spending['ma_khach_hang'].astype(str)

    # Merge membership data back into df_customer
    df_customer = pd.merge(df_customer, customer_spending[['ma_khach_hang', 'membership']], 
                        on='ma_khach_hang', how='left')

    # Fill missing membership values with "Other"
    df_customer['membership'] = df_customer['membership'].fillna("Other")

    #function để recommend theo predict rating_ALS
    def recommend_for_user(ma_khach_hang_moi, new_user_recs, df_asin_asin_idx, df_products_spark, spark):
    
        # Step 1: Filter recommendations for the given user ID
        find_user_rec = new_user_recs.filter(new_user_recs['ma_khach_hang'] == ma_khach_hang_moi).first()

        if not find_user_rec:
            return None  # If no recommendations found

        # Step 2: Extract recommendations (list of structs)
        recommendations = find_user_rec['recommendations']
        rec_rows = [{"ma_san_pham_idx": rec["ma_san_pham_idx"], "predict_rating": rec["rating"]} for rec in recommendations]
        recommendations_df = spark.createDataFrame(rec_rows)

        # Step 3: Join with df_asin_asin_idx to map product indices back to product IDs
        joined_df = recommendations_df.join(
            df_asin_asin_idx,
            recommendations_df["ma_san_pham_idx"] == df_asin_asin_idx["ma_san_pham_idx"],
            "left"
        )

        # Step 4: Select relevant columns (convert to Pandas for merging)
        final_df = joined_df.select(
            col("ma_san_pham"),
            col("predict_rating")
        ).toPandas()

        # Ensure `ma_san_pham` is a string
        final_df["ma_san_pham"] = final_df["ma_san_pham"].astype(str)
        df_products["ma_san_pham"] = df_products["ma_san_pham"].astype(str)

        # Step 5: Merge with df_products (Pandas DataFrame)
        merged_df = final_df.merge(df_products, on="ma_san_pham", how="left")  # Merge with product data
        return merged_df.sort_values(by="predict_rating", ascending=False).head(5)

    
    # Display streamlit
    user_id = st.text_input("Enter User ID", "", key="user_id_input").strip()

    if user_id:
        # Validate user ID in customer database
        matching_rows = df_customer[df_customer["ma_khach_hang"] == user_id]

        if not matching_rows.empty:
            customer_name = matching_rows.iloc[0]["ho_ten"]
            membership_tier = matching_rows.iloc[0]["membership"]
            st.write(f"### Welcome, {customer_name}! :red[{membership_tier}]")

            try:
                # Fetch recommendations
                recommendations = recommend_for_user(
                    ma_khach_hang_moi=user_id,
                    new_user_recs=new_user_recs,
                    df_asin_asin_idx=df_asin_asin_idx,
                    df_products_spark=df_products_spark,
                    spark=spark
                )

                if recommendations is not None:
                    # Display recommended products in a table
                    st.write("### Recommended Products:")
                    st.dataframe(recommendations[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'gia_goc','diem_trung_binh']])

                    # Dropdown for product selection
                    product_options = [
                        (row['ten_san_pham'], row['ma_san_pham'])
                        for _, row in recommendations.iterrows()
                    ]
                    selected_product = st.selectbox(
                        "Chọn sản phẩm",
                        options=product_options,
                        format_func=lambda x: x[0]  # Display product name in dropdown
                    )

                    # Fetch and display selected product's details
                    if selected_product:
                        product_id = selected_product[1]  # Selected product's ID
                        selected_product_details = recommendations[recommendations['ma_san_pham'] == product_id]

                        if not selected_product_details.empty:
                            product_name = selected_product_details['ten_san_pham'].values[0]
                            product_description = selected_product_details['mo_ta'].values[0]
                            truncated_description = ' '.join(product_description.split()[:80])

                            st.write("### Bạn vừa chọn:")
                            st.write(f"**{product_name}**")
                            st.write("##### Thông tin:")
                            st.write(truncated_description, '...')
                            
                    st.image('top10KM.png', use_container_width=True)        
                else: # user mới tạo tài khoản chưa có data thể recommend thì gửi ds top sp
                    st.image('top10KM.png', use_container_width=True)
                    st.image('top_10_best_sellers_updated.png', use_container_width=True)

                    # Tạo một dropdown với options là các tuple này
                    df_products_filtered = df_products[df_products['diem_trung_binh'] >= 4.5]
                    df_products_sorted = df_products_filtered.sort_values(by='ten_san_pham', ascending=True)
                    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in df_products_sorted.iterrows()]
                    selected_product = st.selectbox(
                        "Chọn sản phẩm",
                        options=product_options,
                        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
                        )
                    # Display the selected product
                    st.write("Bạn đã chọn:", selected_product)

                    # Cập nhật session_state dựa trên lựa chọn hiện tại
                    st.session_state.selected_ma_san_pham = selected_product[1]

                    if st.session_state.selected_ma_san_pham:
                        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
                        # Hiển thị thông tin sản phẩm được chọn
                        selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]

                        if not selected_product.empty:
                            st.write('#### Bạn vừa chọn:')
                            st.write('### ', selected_product['ten_san_pham'].values[0])

                            product_description = selected_product['mo_ta'].values[0]
                            truncated_description = ' '.join(product_description.split()[:80])
                            st.write('##### Thông tin:')
                            st.write(truncated_description, '...')

                            # Add Selling Price and Rating
                            selling_price = selected_product['gia_ban'].values[0]  # Get selling price
                            rating = selected_product['diem_trung_binh'].values[0]  # Get rating

                            st.write('**Giá bán:**', f"{selling_price:,} VNĐ")  # Format price with commas
                            st.write('**Đánh giá:**', f"{rating}/5")  # Display rating
            except Exception as e:
                st.error(f"Error fetching recommendations: {str(e)}")
        else:
            st.warning("User ID not found. Please check and try again.")
  

 
