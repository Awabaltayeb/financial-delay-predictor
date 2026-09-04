import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ضبط إعدادات الصفحة لتكون بالاتجاه العربي ومظهر أنيق
st.set_page_config(
    page_title="مساعد التنبؤ بالمتأخرات المالية",
    page_icon="📊",
    layout="centered"
)

# 1. دالة لتوليد بيانات اصطناعية منطقية وتدريب النموذج (يتم تشغيلها مرة واحدة وتخزينها في الذاكرة المؤقتة)
@st.cache_resource
def setup_model_and_data():
    # توليد بيانات اصطناعية (200 عميل) بناءً على قواعد منطقية لضمان تدريب ناجح للنموذج
    np.random.seed(42)
    n_samples = 200
    
    amount = np.random.randint(1000, 5000, n_samples)
    paid = np.array([np.random.randint(0, int(a)) for a in amount])
    previous_delays = np.random.randint(0, 6, n_samples)
    delay_days = np.array([np.random.randint(0, 30) if d > 0 else 0 for d in previous_delays])
    
    # حساب الميزات الإضافية (Feature Engineering)
    paid_ratio = paid / amount
    remaining_amount = amount - paid
    
    # تحديد النتيجة المستهدفة (will_delay) بناءً على وزن منطقي مع إضافة بعض العشوائية (Noise)
    # تزيد احتمالية التأخر إذا كانت نسبة المدفوع ضعيفة، أو التأخيرات السابقة كثيرة
    score = (1.0 - paid_ratio) * 0.4 + (previous_delays / 5.0) * 0.4 + (delay_days / 30.0) * 0.2
    noise = np.random.normal(0, 0.1, n_samples)
    will_delay = np.where(score + noise > 0.5, 1, 0)
    
    # إنشاء DataFrame
    df = pd.DataFrame({
        'amount': amount,
        'paid': paid,
        'previous_delays': previous_delays,
        'delay_days': delay_days,
        'paid_ratio': paid_ratio,
        'remaining_amount': remaining_amount,
        'will_delay': will_delay
    })
    
    # اختيار الميزات المستقلة والمتغير التابع
    features = ['amount', 'paid', 'previous_delays', 'delay_days', 'paid_ratio', 'remaining_amount']
    X = df[features]
    y = df['will_delay']
    
    # تقسيم البيانات للتدريب والاختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # عمل تقييس للبيانات (Scaling) وهو أمر أساسي للـ Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # تدريب النموذج
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # قياس الدقة
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, df, features

# تشغيل دالة الإعداد والحصول على النموذج والبيانات
model, scaler, accuracy, df, features_list = setup_model_and_data()

# --- واجهة المستخدم (Streamlit UI) ---

# رأس الصفحة
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>مساعد الشؤون المالية للتنبؤ بالمتأخرات 📊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>نظام ذكي مبسط لمساعدة موظف المالية في التنبؤ باحتمالية تأخر العملاء عن السداد باستخدام الذكاء الاصطناعي.</p>", unsafe_allow_html=True)
st.write("---")

# الحقول المدخلة من قبل المستخدم
st.subheader("📝 إدخال بيانات العميل")

col1, col2 = st.columns(2)

with col1:
    amount_input = st.number_input("المبلغ الإجمالي المطلوب ($)", min_value=100, max_value=10000, value=2000, step=100)
    paid_input = st.number_input("المبلغ المدفوع حتى الآن ($)", min_value=0, max_value=amount_input, value=800, step=50)

with col2:
    prev_delays_input = st.slider("عدد مرات التأخر السابقة", min_value=0, max_value=5, value=1)
    prev_delay_days_input = st.slider("مجموع أيام التأخر السابقة", min_value=0, max_value=30, value=5)

st.write("")

# زر التنبؤ والتوقع
if st.button("تحليل وتوقع النتيجة 🔍", use_container_width=True):
    # 1. حساب الميزات الإضافية المدخلة
    p_ratio = paid_input / amount_input
    rem_amount = amount_input - paid_input
    
    # 2. تجهيز البيانات كـ DataFrame لتطابق ميزات التدريب
    input_data = pd.DataFrame([{
        'amount': amount_input,
        'paid': paid_input,
        'previous_delays': prev_delays_input,
        'delay_days': prev_delay_days_input,
        'paid_ratio': p_ratio,
        'remaining_amount': rem_amount
    }])
    
    # 3. عمل تقييس للمدخلات باستخدام الـ Scaler المدرب سابقاً
    input_scaled = scaler.transform(input_data[features_list])
    
    # 4. التنبؤ بالنتيجة واحتماليتها
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0] # يعيد مصفوفة [احتمال 0, احتمال 1]
    
    delay_probability = probabilities[1] * 100
    commit_probability = probabilities[0] * 100
    
    st.write("---")
    st.subheader("🎯 نتيجة التنبؤ بالذكاء الاصطناعي")
    
    if prediction == 1:
        st.warning("⚠️ **النتيجة المتوقعة:** العميل غالباً **سيتأخر** في السداد.")
        st.metric(label="نسبة احتمال التأخر في السداد", value=f"{delay_probability:.1f}%")
        st.info("💡 **توصية:** يفضل التواصل مع العميل بشكل استباقي لتأكيد موعد الدفع أو طلب ضمانات إضافية.")
    else:
        st.success("✅ **النتيجة المتوقعة:** العميل غالباً **سيلتزم** بالسداد في الوقت المحدد.")
        st.metric(label="نسبة احتمال الالتزام في السداد", value=f"{commit_probability:.1f}%")
        st.info("💡 **توصية:** العميل يظهر مؤشرات التزام جيدة بناءً على سجله التاريخي ونسبة دفعه الحالية.")

st.write("---")

# قسم أكاديمي إضافي مفيد جداً للمناقشة مع المشرفين
with st.expander("🛠️ تفاصيل تدريب النموذج والبيانات (مخصص للمناقشة والأستاذ)"):
    st.write("هذا القسم مخصص لإظهار كيف يعمل الذكاء الاصطناعي في الخلفية:")
    
    # عرض دقة النموذج المقياسة على بيانات الاختبار
    st.write(f"📈 **دقة النموذج الحالية (Accuracy):** `{accuracy * 100:.1f}%` (تم اختباره على بيانات غير مرئية)")
    st.write("⚙️ **الخوارزمية المستخدمة:** `Logistic Regression` (الانحدار اللوجستي للتصنيف الثنائي).")
    
    # عرض عينة من البيانات الاصطناعية التي تدرب عليها
    st.write("📋 **عينة من البيانات الاصطناعية المستخدمة في تدريب النموذج (أول 5 أسطر):**")
    st.dataframe(df.head(5))
    st.caption("ملاحظة: البيانات المولدة اصطناعية لغرض التدريب، ولكنها تتبع سلوكاً مالياً منطقياً.")
