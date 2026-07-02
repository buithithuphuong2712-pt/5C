import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Thuật toán mặc định phù hợp cho phân loại nhãn nhị phân
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIG (Bắt buộc là lệnh đầu tiên)
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Dự Báo Tác Động - Machine Learning App",
    page_icon="🔮"
)

# -----------------------------------------------------------------------------
# STEP 2: CACHED DATA LOADING FUNCTION
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để tối ưu hóa caching của Streamlit.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa chính xác tập biến đầu vào và mục tiêu trích xuất từ Notebook
FEATURE_COLS = [
    'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
    'NL1', 'NL2', 'NL3', 'NL4', 
    'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
    'TS1', 'TS2', 'TS3', 'TS4'
]
TARGET_COL = 'PD'

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR - CONFIGURATION ZONE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file 5c.csv hoặc cấu trúc tương tự có chứa các cột đặc trưng và biến mục tiêu 'PD'."
    )
    
    st.divider()
    
    st.subheader("Tham số mô hình AI")
    # Sử dụng thuật toán RandomForest ngầm định làm nền tảng phân loại dữ liệu bảng từ khảo sát
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong mô hình rừng ngẫu nhiên."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, max_value=30, value=10, step=1,
        help="Độ sâu giới hạn của mỗi cây quyết định giúp hạn chế Overfitting."
    )
    
    test_size = st.slider(
        "Tỷ lệ tập kiểm định (test_size)", 
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="Tỷ lệ chia dữ liệu để đánh giá mô hình. Mặc định là 0.2 (20%)."
    )
    
    random_state = st.number_input(
        "Hạt giống ngẫu nhiên (random_state)", 
        value=23, step=1,
        help="Giá trị cố định để đảm bảo kết quả phân tách dữ liệu tái lập chính xác giống Notebook (mặc định: 23)."
    )
    
    st.divider()
    
    # Điểm kích hoạt huấn luyện duy nhất
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# STEP 4: HEADER ZONE
# -----------------------------------------------------------------------------
st.title("🔮 Ứng dụng Dự Báo Phân Loại Chỉ Số Đối Tượng")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu, trực quan hóa đặc trưng khảo sát và xây dựng mô hình dự báo biến mục tiêu (PD) tự động dựa trên quy trình Machine Learning.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên file dữ liệu huấn luyện mẫu (ví dụ: `5c.csv`) ở Sidebar để bắt đầu kích hoạt hệ thống.")
    st.stop()

# Đọc dữ liệu qua hàm cache chung
file_bytes = uploaded_file.getvalue()
df_main = load_data(file_bytes, uploaded_file.name)

if df_main is not None:
    st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Quy mô dữ liệu thô: `{df_main.shape[0]}` dòng, `{df_main.shape[1]}` cột.")
st.divider()

# -----------------------------------------------------------------------------
# STEP 5: MODEL TRAINING LOGIC (Chạy khi bấm nút, lưu kết quả vào session_state)
# -----------------------------------------------------------------------------
if btn_train:
    with st.spinner("⏳ Hệ thống đang tiến hành xử lý dữ liệu và huấn luyện mô hình AI..."):
        # Kiểm tra sự tồn tại của các trường thông tin bắt buộc
        missing_features = [col for col in FEATURE_COLS if col not in df_main.columns]
        
        if missing_features:
            st.error(f"❌ Dữ liệu tải lên thiếu các cột biến đầu vào sau: {missing_features}")
        elif TARGET_COL not in df_main.columns:
            st.error(f"❌ Không tìm thấy cột biến mục tiêu '{TARGET_COL}' trong dữ liệu.")
        else:
            # Tách X, y
            X = df_main[FEATURE_COLS]
            y = df_main[TARGET_COL]
            
            # Chia tập dữ liệu tương tự thiết lập từ Notebook
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Xây dựng và khớp mô hình
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state
            )
            model.fit(X_train, y_train)
            
            # Đánh giá và lưu trữ kết quả kiểm định
            y_pred = model.predict(X_test)
            
            # Lưu trữ trạng thái vào st.session_state
            st.session_state['trained_model'] = model
            st.session_state['feature_columns'] = FEATURE_COLS
            st.session_state['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', fallback=0),
                'recall': recall_score(y_test, y_pred, average='binary', fallback=0),
                'f1': f1_score(y_test, y_pred, average='binary', fallback=0),
                'cm': confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển qua các tab bên dưới để xem kết quả chi tiết.")

# -----------------------------------------------------------------------------
# STEP 6: MAIN TABS SYSTEM
# -----------------------------------------------------------------------------
tab_summary, tab_viz, tab_metrics, tab_predict = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab_summary:
    st.subheader("Thống kê cấu trúc tệp tin")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi (Dòng)", df_main.shape[0])
    col2.metric("Tổng số cột đặc trưng", df_main.shape[1])
    col3.metric("Kích thước tệp tin", f"{len(file_bytes)/(1024*1024):.4f} MB")
    
    st.markdown("### Xem trước dữ liệu thô (5 dòng đầu tiên)")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    st.markdown("### Thống kê mô tả các biến mô hình (X & y)")
    # Chỉ mô tả các biến đưa vào mô hình để giao diện không bị loãng
    available_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df_main.columns]
    st.dataframe(df_main[available_cols].describe(), use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab_viz:
    st.subheader("Biểu đồ phân phối các biến đặc trưng chính")
    
    # Ưu tiên biến mục tiêu y (PD) đầu tiên nếu có trong dữ liệu
    if TARGET_COL in df_main.columns:
        fig_target = px.histogram(
            df_main, x=TARGET_COL, 
            title=f"Phân phối tần suất của Biến mục tiêu: {TARGET_COL}",
            color=TARGET_COL, text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_target.update_layout(height=350)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Khám phá tương quan các nhóm câu hỏi đặc trưng đầu vào (TC, NL, DK, V, TS)")
    
    # Cho phép người dùng đa chọn linh hoạt các biến muốn xem biểu đồ dạng lưới 2x2
    selected_features = st.multiselect(
        "Chọn nhóm các biến đầu vào để hiển thị phân phối phân tích (Mặc định chọn 4 đặc trưng đại diện):",
        options=FEATURE_COLS,
        default=FEATURE_COLS[:4]
    )
    
    if selected_features:
        # Bố trí biểu đồ lưới 2 cột cân đối
        grid_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            current_col = grid_cols[idx % 2]
            if col_name in df_main.columns:
                fig_feat = px.histogram(
                    df_main, x=col_name,
                    title=f"Phân phối điểm số của chỉ số: {col_name}",
                    text_auto=True,
                    color_discrete_sequence=['#4A90E2']
                )
                fig_feat.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                current_col.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Vui lòng lựa chọn ít nhất một biến đặc trưng từ danh sách để vẽ biểu đồ trực quan.")

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab_metrics:
    st.subheader("Đánh giá hiệu năng mô hình phân loại dự báo")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Chưa tìm thấy mô hình đã huấn luyện. Vui lòng thiết lập các tham số ở Sidebar và nhấn nút **'Huấn luyện mô hình'**.")
    else:
        metrics = st.session_state['metrics']
        
        # Hiển thị các chỉ tiêu dạng số lớn scannable
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        m_col2.metric("Độ chính xác dự báo đúng (Precision)", f"{metrics['precision']:.4f}")
        m_col3.metric("Tỷ lệ bỏ sót (Recall)", f"{metrics['recall']:.4f}")
        m_col4.metric("Điểm F1-Score cân bằng", f"{metrics['f1']:.4f}")
        
        st.markdown("---")
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("#### 🟥 Ma trận nhầm lẫn (Confusion Matrix)")
            cm_df = pd.DataFrame(
                metrics['cm'], 
                index=['Thực tế: 0', 'Thực tế: 1'], 
                columns=['Dự báo: 0', 'Dự báo: 1']
            )
            st.dataframe(cm_df, use_container_width=True)
            
            # Trực quan hóa ma trận nhầm lẫn bằng heatmap
            fig_cm = px.imshow(
                metrics['cm'],
                text_auto=True,
                labels=dict(x="Nhãn Dự Báo", y="Nhãn Thực Tế"),
                x=['0 (Sai/Không)', '1 (Đúng/Có)'],
                y=['0 (Sai/Không)', '1 (Đúng/Có)'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=280)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with c_right:
            st.markdown("#### 📋 Báo cáo phân loại chi tiết (Classification Report)")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH (INFERENCE ZONE) ---
with tab_predict:
    st.subheader("Chẩn đoán & Dự báo tương tác")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Tính năng này yêu cầu mô hình phải được huấn luyện trước. Vui lòng bấm nút kích hoạt ở Sidebar.")
    else:
        model = st.session_state['trained_model']
        
        mode = st.radio(
            "Chọn hình thức kiểm tra đầu vào mô hình:",
            ["Chế độ 1: Nhập dữ liệu trực tiếp thủ công", "Chế độ 2: Tải file dữ liệu danh sách kiểm tra hàng loạt"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        if mode == "Chế độ 1: Nhập dữ liệu trực tiếp thủ công":
            st.markdown("👉 *Vui lòng điền thông số điểm số của bảng hỏi (từ 1 đến 5 hoặc tương đương) dưới đây:*")
            
            with st.form("single_prediction_form"):
                # Gom gọn giao diện thành hệ thống lưới cột để tránh cuộn trang quá dài
                form_cols = st.columns(4)
                input_data = {}
                
                for idx, col_name in enumerate(FEATURE_COLS):
                    target_container = form_cols[idx % 4]
                    # Lấy giá trị mặc định dựa trên dữ liệu lịch sử (trung vị hoặc trung bình) giúp điền form nhanh
                    default_val = float(df_main[col_name].median()) if col_name in df_main.columns else 3.0
                    
                    input_data[col_name] = target_container.number_input(
                        f"Chỉ số {col_name}",
                        min_value=0.0, max_value=10.0, value=default_val, step=1.0,
                        help=f"Nhập thông số điểm của thuộc tính {col_name}"
                    )
                
                submit_pred = st.form_submit_button("🔮 Chẩn đoán kết quả", type="primary", use_container_width=True)
                
                if submit_pred:
                    # Chuyển đổi dữ liệu sang định dạng DataFrame cấu trúc chuẩn
                    input_df = pd.DataFrame([input_data])
                    single_pred = model.predict(input_df)[0]
                    
                    st.markdown("### Kết quả dự báo của hệ thống:")
                    if single_pred == 1:
                        st.success(f"🎯 Kết quả chẩn đoán: **PD = {single_pred} (Tích cực / Đạt chỉ tiêu)**")
                    else:
                        st.warning(f"⚠️ Kết quả chẩn đoán: **PD = {single_pred} (Hạn chế / Chưa đạt chỉ tiêu)**")
                    
                    # Hiện xác suất nếu mô hình hỗ trợ
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(input_df)[0]
                        st.caption(f"Xác suất dự đoán lớp: Phân lớp 0 (`{prob[0]*100:.2f}%`) | Phân lớp 1 (`{prob[1]*100:.2f}%`)")

        # CHẾ ĐỘ 2 — TẢI FILE HÀNG LOẠT
        else:
            st.markdown("👉 *Tải lên file dữ liệu mới cần dự báo. Định dạng cấu trúc file yêu cầu chứa đủ các cột thuộc tính đặc trưng đầu vào.*")
            bulk_file = st.file_uploader("Tải tệp kiểm tra hàng loạt (.csv)", type=["csv"], key="bulk_upload")
            
            if bulk_file is not None:
                bulk_df = pd.read_csv(bulk_file)
                missing_cols = [c for c in FEATURE_COLS if c not in bulk_df.columns]
                
                if missing_cols:
                    st.error(f"❌ File tải lên không đúng cấu trúc schema mong đợi. Thiếu các cột sau: {missing_cols}")
                else:
                    # Lọc lấy đúng tập đặc trưng phục vụ chạy mô hình
                    X_bulk = bulk_df[FEATURE_COLS]
                    bulk_preds = model.predict(X_bulk)
                    
                    # Gắn kết quả dự báo mới vào file xuất dữ liệu
                    bulk_df['PD_Dự_Báo'] = bulk_preds
                    
                    if hasattr(model, "predict_proba"):
                        bulk_probs = model.predict_proba(X_bulk)
                        bulk_df['Xác_Suất_Lớp_1'] = bulk_probs[:, 1]
                    
                    st.success(f"✅ Dự báo thành công cho `{len(bulk_df)}` dòng dữ liệu bản ghi mới!")
                    st.markdown("#### Bảng dữ liệu kết quả phân lớp chi tiết:")
                    st.dataframe(bulk_df, use_container_width=True)
                    
                    # Chuyển đổi DataFrame kết quả để tạo nút Download
                    csv_buffer = bulk_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống tệp kết quả dự báo (CSV)",
                        data=csv_buffer,
                        file_name="ket_qua_du_bao_hang_loat.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
