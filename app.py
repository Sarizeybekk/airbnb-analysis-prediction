import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap, MarkerCluster
import tempfile
import os

def folium_static(fig, height=500):
    """Render folium map by saving to HTML."""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_path = temp.name
    temp.close()
    fig.save(temp_path)
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    os.unlink(temp_path)
    components.html(html_data, height=height)


st.set_page_config(page_title="Airbnb Fiyat Tahmini", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5A5F; /* Airbnb color */
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #484848;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #767676;
        margin-top: 1rem;
    }
    .description {
        font-size: 1rem;
        color: #484848;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF5A5F;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigasyon")
pages = ["Ana Sayfa", "Veri İnceleme", "Ön İşleme Sonuçları", "Model Sonuçları", "Harita Görselleştirme"]
selected_page = st.sidebar.radio("", pages)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("AB_NYC_2019.csv")
        return df
    except:
        st.error("Lütfen AB_NYC_2019.csv dosyasını yükleyin veya doğru konumda olduğundan emin olun.")
        return None

df = load_data()

if df is None:
    st.warning("Devam etmek için veri dosyasını yükleyin.")
    uploaded_file = st.file_uploader("AB_NYC_2019.csv dosyasını yükleyin", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Veri başarıyla yüklendi!")

def preprocess_data(df):
   
    processed_df = df.copy()
   
    processed_df.drop(columns=["id", "name", "host_id", "host_name", "last_review"], inplace=True)
 
    power_transformer = PowerTransformer(method='yeo-johnson')
    reviews_temp = processed_df["reviews_per_month"].fillna(0)
    processed_df["reviews_per_month"] = power_transformer.fit_transform(reviews_temp.values.reshape(-1, 1))

    processed_df["reviews_per_month_original"] = power_transformer.inverse_transform(
        processed_df["reviews_per_month"].values.reshape(-1, 1)
    ).flatten()

    processed_df = pd.get_dummies(processed_df, columns=["neighbourhood_group", "room_type"], drop_first=True)
  
    processed_df["neighbourhood_encoded"] = processed_df.groupby("neighbourhood")["price"].transform("mean")
    processed_df.drop(columns=["neighbourhood"], inplace=True)
    
    processed_df = processed_df[processed_df["price"] > 0]

    processed_df["log_price"] = np.log1p(processed_df["price"])
    processed_df["minimum_nights_log"] = np.log1p(processed_df["minimum_nights"])
    processed_df["review_score"] = processed_df["reviews_per_month"] * processed_df["number_of_reviews"]
   
    X = processed_df.drop(columns=["price"])
    if "reviews_per_month_original" in X.columns:
        X = X.drop(columns=["reviews_per_month_original"])
    y = processed_df["price"]
    
    return X, y, processed_df

if df is not None:

    if selected_page == "Ana Sayfa":
        st.markdown("<h1 class='main-header'>New York Airbnb Fiyat Tahmini</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("https://a0.muscache.com/im/pictures/fe7217ff-0b24-438d-8833-1dd45a298a6b.jpg", use_column_width=True)
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        Bu veri seti ile New York City'deki Airbnb ilanlarının fiyatlarını tahmin etmeye çalışacağız. 
        Amacımız, bir evi kiralamak isteyen birinin fiyatının ne olacağını öngörmek. 
        Bunun için Regresyon, Karar Ağacı ve Random Forest modelleri uygulayacağız.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-header'>Veri Seti Genel Bakış</h2>", unsafe_allow_html=True)
        st.dataframe(df.head())
        
        st.markdown("<h2 class='section-header'>Veri Seti Sütunları</h2>", unsafe_allow_html=True)
        st.markdown("""
        - **id**: Airbnb ilanının benzersiz kimlik numarası
        - **name**: İlanın adı veya açıklaması
        - **host_id**: İlan sahibinin benzersiz kimlik numarası
        - **host_name**: İlan sahibinin adı
        - **neighbourhood_group**: İlanın bulunduğu büyük bölge (örneğin Manhattan, Brooklyn)
        - **neighbourhood**: İlanın bulunduğu mahalle
        - **latitude**: İlanın enlem (latitude) koordinatı
        - **longitude**: İlanın boylam (longitude) koordinatı
        - **room_type**: Konaklama türü (Örneğin: "Private room", "Entire home/apt", "Shared room")
        - **price**: Gecelik konaklama ücreti (USD cinsinden)
        - **minimum_nights**: Konaklama için belirlenen minimum gece sayısı
        - **number_of_reviews**: İlanın aldığı toplam inceleme sayısı
        - **last_review**: İlanın son inceleme tarihi
        - **reviews_per_month**: Aylık ortalama inceleme sayısı
        - **calculated_host_listings_count**: Aynı ev sahibinin toplam ilan sayısı
        - **availability_365**: Yıl boyunca müsait olduğu gün sayısı (365 gün üzerinden)
        """)
    
  
    elif selected_page == "Veri İnceleme":
        st.markdown("<h1 class='main-header'>Veri İnceleme</h1>", unsafe_allow_html=True)
     
        st.markdown("<h2 class='section-header'>Aylık Yorum Sayısı Analizi</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ortalama", f"{df['reviews_per_month'].mean():.2f}")
        with col2:
            st.metric("Medyan", f"{df['reviews_per_month'].median():.2f}")
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        Çarpık bir dağılım var. Çünkü ortalama > medyan olduğundan dağılım sağa çarpık. 
        Bu da birçok ilan çok az yorum alırken, az sayıda ilan aşırı fazla yorum aldığını gösteriyor.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["reviews_per_month"].dropna(), bins=50, kde=True, ax=ax)
        ax.axvline(df["reviews_per_month"].dropna().mean(), color='red', linestyle='dashed', linewidth=2, label="ortalama")
        ax.axvline(df["reviews_per_month"].dropna().median(), color='blue', linestyle='dashed', linewidth=2, label="median")
        ax.set_title("Aylık Yorum Sayısı Dağılımı")
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown("""
        Daha fazla yorum sayısına sahip ilanlar ortalamayı yukarı çekiyor. 
        Burada çarpıklığı azaltmam gerekiyor. Dağılımı dengeli hale getirelim.
        """)

        st.markdown("<h2 class='section-header'>Eksik Değer Analizi</h2>", unsafe_allow_html=True)
        
        missing_vals = df.isnull().sum()
        missing_cols = missing_vals[missing_vals > 0]
        
        if len(missing_cols) > 0:
            st.write("Eksik değer içeren sütunlar:")
            st.write(missing_cols)
        else:
            st.success("Veri setinde eksik değer bulunmuyor.")
    
 
    elif selected_page == "Ön İşleme Sonuçları":
        st.markdown("<h1 class='main-header'>Ön İşleme Sonuçları</h1>", unsafe_allow_html=True)
        
  
        st.markdown("<h2 class='section-header'>Yeo-Johnson Dönüşümü Sonuçları</h2>", unsafe_allow_html=True)
   
        processed_df = df.copy()
  
        try:
            power_transformer = PowerTransformer(method='yeo-johnson')
            reviews_temp = processed_df["reviews_per_month"].fillna(0)
            reviews_transformed = power_transformer.fit_transform(reviews_temp.values.reshape(-1, 1))
            reviews_original = power_transformer.inverse_transform(reviews_transformed)
            
    
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            
  
            sns.histplot(reviews_temp, bins=50, kde=True, ax=axes[0], color='skyblue')
            axes[0].axvline(reviews_temp.mean(), color='red', linestyle='dashed', linewidth=2, label="ortalama")
            axes[0].axvline(reviews_temp.median(), color='blue', linestyle='dashed', linewidth=2, label="median")
            axes[0].set_title("Orijinal Veride Aylık Yorum Dağılımı")
            axes[0].legend()
            
         
            sns.histplot(reviews_transformed, bins=50, kde=True, ax=axes[1], color='lightgreen')
            axes[1].axvline(reviews_transformed.mean(), color='red', linestyle='dashed', linewidth=2, label="ortalama")
            axes[1].axvline(np.median(reviews_transformed), color='blue', linestyle='dashed', linewidth=2, label="median")
            axes[1].set_title("Yeo-Johnson Dönüşüm Sonrası Dağılım")
            axes[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
          
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
          
            sns.boxplot(x=reviews_transformed.flatten(), ax=ax[0])
            ax[0].set_title("Aylık Yorum Sayısı Dağılımı (Boxplot)")
            
         
            sns.histplot(reviews_transformed.flatten(), bins=30, kde=True, ax=ax[1])
            ax[1].set_title("Aylık Yorum Sayısı Histogramı")
            
            st.pyplot(fig)
            
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("""
            ✓ Çünkü veri uç değerler içeriyor ve çarpık bir dağılım gösteriyor.  
            ✓ Çoğu ilan çok az yorum alırken, birkaç ilan çok fazla yorum alıyor.  
            ✓ mean() kullanırsak, az yorum alan ilanlar için yanlış tahmin yapabiliriz.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Dönüşüm sırasında hata oluştu: {e}")
        
  
        st.markdown("<h2 class='section-header'>Eksik Değer Doldurma Stratejisi</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        Eksik değerleri şu stratejiye göre doldurdum:
        
        1. Önce reviews_per_month sütununu float tipine çevirdim.
        2. Aynı mahalle ve oda tipindeki medyan değerleri kullanarak eksik değerleri doldurdum.
        3. Hala eksik değer varsa, mahalle bazında medyan değerleri kullandım.
        4. Son olarak, kalan eksik değerleri 0 ile doldurdum.
        """)
        
 
        st.markdown("<h2 class='section-header'>Kategorik Değişken Dönüşümü ve Özellik Mühendisliği</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        1. neighbourhood_group ve room_type kategorik değişkenlerini one-hot encoding ile sayısal değerlere dönüştürdüm.
        2. neighbourhood değişkenini, her mahalle için ortalama fiyatı hesaplayarak kodladım.
        3. Sıfır fiyatlı ilanları veri setinden çıkardım.
        4. Logaritmik dönüşümler uygulayarak log_price ve minimum_nights_log değişkenlerini oluşturdum.
        5. reviews_per_month ve number_of_reviews değişkenlerini birleştirerek review_score adlı yeni bir özellik oluşturdum.
        """)
    
   
    elif selected_page == "Model Sonuçları":
        st.markdown("<h1 class='main-header'>Model Sonuçları</h1>", unsafe_allow_html=True)
        
  
        if st.checkbox("Modelleri Göster", value=True):
            with st.spinner("Modeller hazırlanıyor..."):
               
                X, y, processed_df = preprocess_data(df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
              
                st.markdown("<h2 class='section-header'>Doğrusal Regresyon Sonuçları</h2>", unsafe_allow_html=True)
                
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_pred = lr_model.predict(X_test)
                
                lr_mae = mean_absolute_error(y_test, lr_pred)
                lr_mse = mean_squared_error(y_test, lr_pred)
                lr_rmse = np.sqrt(lr_mse)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${lr_mae:.2f}")
                col2.metric("RMSE", f"${lr_rmse:.2f}")
                col3.metric("MSE", f"${lr_mse:.2f}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=lr_pred, alpha=0.5, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
                ax.set_xlabel("Gerçek Değerler (Fiyat)")
                ax.set_ylabel("Tahmin Edilen Değerler (Fiyat)")
                ax.set_title("Gerçek vs. Tahmin Edilen Fiyatlar")
                
          
                
                st.pyplot(fig)
                
         
                st.markdown("<h2 class='section-header'>Karar Ağacı Sonuçları</h2>", unsafe_allow_html=True)
                
                dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
                dt_model.fit(X_train, y_train)
                dt_pred = dt_model.predict(X_test)
                
                dt_mae = mean_absolute_error(y_test, dt_pred)
                dt_mse = mean_squared_error(y_test, dt_pred)
                dt_r2 = r2_score(y_test, dt_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${dt_mae:.2f}")
                col2.metric("MSE", f"${dt_mse:.2f}")
                col3.metric("R²", f"{dt_r2:.4f}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=dt_pred, alpha=0.5, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="brown", linestyle="--")
                ax.set_xlabel("Gerçek Değerler (Fiyat)")
                ax.set_ylabel("Tahmin Edilen Değerler (Fiyat)")
                ax.set_title("Gerçek vs. Tahmin Edilen Fiyatlar")
                
            
                
                st.pyplot(fig)
                
                if st.checkbox("Karar Ağacını Görselleştir"):
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, ax=ax)
                    ax.set_title("Decision Tree")
                    st.pyplot(fig)
                

                st.markdown("<h2 class='section-header'>Random Forest Sonuçları</h2>", unsafe_allow_html=True)
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                
                rf_mae = mean_absolute_error(y_test, rf_pred)
                rf_mse = mean_squared_error(y_test, rf_pred)
                rf_r2 = r2_score(y_test, rf_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${rf_mae:.2f}")
                col2.metric("MSE", f"${rf_mse:.2f}")
                col3.metric("R²", f"{rf_r2:.4f}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=rf_pred, alpha=0.5, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
                ax.set_xlabel("Gerçek Değerler (Fiyat)")
                ax.set_ylabel("Tahmin Edilen Değerler (Fiyat)")
                ax.set_title("Gerçek vs. Tahmin Edilen Fiyatlar")
                
        
                
                st.pyplot(fig)
        
                st.markdown("<h2 class='section-header'>Model Karşılaştırması</h2>", unsafe_allow_html=True)
                
                comparison_df = pd.DataFrame({
                    'Model': ['Doğrusal Regresyon', 'Karar Ağacı', 'Random Forest'],
                    'MAE': [lr_mae, dt_mae, rf_mae],
                    'MSE': [lr_mse, dt_mse, rf_mse],
                    'R²': [r2_score(y_test, lr_pred), dt_r2, rf_r2]
                })
                
                comparison_df = comparison_df.set_index('Model')
                st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'MSE']).highlight_max(subset=['R²']))
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("""
                Random Forest modeli en iyi performansı göstermiştir. 
                R² değeri 1'e yakın olduğu için modelin açıklama gücü yüksektir.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
    

    elif selected_page == "Harita Görselleştirme":
        st.markdown("<h1 class='main-header'>Harita Görselleştirme</h1>", unsafe_allow_html=True)
        
        try:
    
            st.markdown("<h2 class='section-header'>Semtlere Göre Konum Dağılımı</h2>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df.longitude, y=df.latitude, hue=df.neighbourhood_group, ax=ax)
            ax.set_title('Neighbourhood Group Location')
            st.pyplot(fig)
            
     
            st.markdown("<h2 class='section-header'>Oda Tiplerine Göre Konum Dağılımı</h2>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df.longitude, y=df.latitude, hue=df.room_type, ax=ax)
            ax.set_title('Room type location per Neighbourhood Group')
            st.pyplot(fig)
            
           
            st.markdown("<h2 class='section-header'>İlan Yoğunluğu Haritası</h2>", unsafe_allow_html=True)
            
            if st.button("Yoğunluk Haritasını Göster"):
               
                m = folium.Map(location=[40.76586, -73.98436], tiles='cartodbpositron', zoom_start=11)
                
         
                sample_df = df.sample(min(5000, len(df)))
                
               
                HeatMap(data=sample_df[['latitude', 'longitude']].values.tolist(), radius=10).add_to(m)
                
            
                folium_static(m, height=600)
            
      
            st.markdown("<h2 class='section-header'>İlan Kümeleme Haritası</h2>", unsafe_allow_html=True)
            
            if st.button("Kümeleme Haritasını Göster"):
             
                m = folium.Map(location=[40.76586, -73.98436], tiles='cartodbpositron', zoom_start=11)
          
                sample_df = df.sample(min(1000, len(df)))
                
         
                sample_df["All"] = 'Room type: ' + sample_df['room_type'].astype(str) + ', ' + \
                                'Availability (365 days): ' + sample_df["availability_365"].astype(str) + ', ' + \
                                'Price: $' + sample_df["price"].astype(str)
                
       
                marker_cluster = MarkerCluster().add_to(m)
                
        
                for idx, row in sample_df.iterrows():
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=row['All']
                    ).add_to(marker_cluster)
            
                folium_static(m, height=600)
        except Exception as e:
            st.error(f"Harita oluşturulurken bir hata oluştu: {e}")

if __name__ == "__main__":
   
    pass