import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ضبط إعدادات الصفحة لتكون بالاتجاه العربي ومظهر أنيق مخصص للكلية
st.set_page_config(
    page_title="نظام التنبؤ الذكي بالمتأخرات الدراسية",
    page_icon="🎓",
    layout="centered"
)

# 1. دالة لتوليد بيانات لـ 1000 طالب منطقية وتدريب نموذج Random Forest
@st.cache_resource
def setup_rf_model():
    # توليد بيانات اصطناعية لـ 1000 طالب بناءً على سلوك سداد الرسوم الدراسية المذكور بالخطة
    np.random.seed(42)
    n_samples = 1000
    
    # 1. الدخل الشهري للأسرة (قيم مرنة وواقعية لبيئات مالية مختلفة)
    family_income = np.random.randint(10000, 150000, n_samples) 
    # 2. نسبة المنحة الدراسية (من 0% إلى 100%)
    scholarship_ratio = np.random.uniform(0.0, 1.0, n_samples) 
    # 3. سجل التأخيرات السابقة (عدد مرات التأخر من 0 إلى 5)
    previous_delays = np.random.randint(0, 6, n_samples) 
    
    # حساب مصفوفة القرار لإنتاج نتيجة مستهدفة (will_delay) منطقية جداً:
    # تزيد احتمالية التأخر عند انخفاض الدخل، وانخفاض المنحة، وارتفاع التأخيرات السابقة
    income_score = (150000 - family_income) / 140000  # قيمة من 0 (دخل عالي، خطورة منخفضة) إلى 1 (دخل منخفض، خطورة عالية)
    scholarship_score = 1.0 - scholarship_ratio       # قيمة من 0 (منحة كاملة، خطورة منخفضة) إلى 1 (لا توجد منحة، خطورة عالية)
    delays_score = previous_delays / 5.0              # قيمة من 0 إلى 1
    
    combined_score = (income_score * 0.4) + (scholarship_score * 0.3) + (delays_score * 0.3)
    noise = np.random.normal(0, 0.08, n_samples) # إضافة بعض العشوائية الواقعية للبيانات
    
    will_delay = np.where(combined_score + noise > 0.52, 1, 0)
    
    # إنشاء DataFrame
    df = pd.DataFrame({
        'family_income': family_income,
        'scholarship_ratio': scholarship_ratio * 100,  # تحويل النسبة لمئوية لتسهيل العرض
        'previous_delays': previous_delays,
        'will_delay': will_delay
    })
    
    # تحديد الميزات لتدريب الخوارزمية
    features = ['family_income', 'scholarship_ratio', 'previous_delays']
    X = df[features]
    y = df['will_delay']
    
    # تقسيم البيانات (80% تدريب، 20% اختبار لتقييم الدقة)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تدريب نموذج الغابة العشوائية (Random Forest Classifier)
    # ملاحظة: هذه الخوارزمية لا تحتاج لتقييس البيانات (Scaling) وتتعلم بكفاءة عالية جداً
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # قياس الدقة
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # الحصول على أهمية المتغيرات (Feature Importance)
    importances = model.feature_importances_
    
    return model, accuracy, df, features, importances

# تشغيل التدريب والحصول على "عقل النظام"
model, accuracy, df, features_list, importances = setup_rf_model()

# --- واجهة المستخدم العصرية (Streamlit UI) ---

# رأس الصفحة بناءً على المسمى الرسمي المعتمد في خطتك المفقودة
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>نظام التنبؤ الذكي بالمتأخرات الدراسية 🎓</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #3B82F6;'>Smart Fees Prediction System</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>نظام ريادي مدعوم بالذكاء الاصطناعي لاستشراف الأزمات المالية للطلاب ومساعدة الشؤون المالية بالكلية على اتخاذ قرارات استباقية.</p>", unsafe_allow_html=True)
st.write("---")

# واجهة إدخال البيانات
st.subheader("📝 إدخال البيانات التحليلية للطالب")

# خانة اسم الطالب لإعطاء مظهر رسمي للنظام
student_name = st.text_input("اسم الطالب بالكامل", value="أحمد محمد علي")

col1, col2 = st.columns(2)

with col1:
    # إدخال حر للدخل الشهري دون أي قيود تمنع الأرقام الكبيرة
    family_income_input = st.number_input("الدخل الشهري لأسرة الطالب", min_value=0, value=45000, step=1000)
    scholarship_ratio_input = st.slider("نسبة المنحة الدراسية التي يتلقاها الطالب (%)", min_value=0, max_value=100, value=25)

with col2:
    prev_delays_input = st.slider("سجل التأخيرات السابقة (عدد الفصول الدراسية الماضية التي تأخر فيها العميل)", min_value=0, max_value=5, value=1)
    st.info("💡 **تحليل رقمي:** يتم حساب نسبة التعثر تلقائياً بالدمج بين سلوك السداد وقدرة الأسرة الائتمانية.")

st.write("")

# زر التنبؤ الفوري
if st.button("تحليل وتوقع النتيجة الفورية 🔍", use_container_width=True):
    # تجهيز المدخلات
    input_data = pd.DataFrame([{
        'family_income': family_income_input,
        'scholarship_ratio': scholarship_ratio_input,
        'previous_delays': prev_delays_input
    }])
    
    # التنبؤ المباشر باستخدام الغابة العشوائية
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    delay_probability = probabilities[1] * 100
    commit_probability = probabilities[0] * 100
    
    st.write("---")
    st.subheader(f"🎯 تقرير الحالة المتوقعة للطالب: {student_name}")
    
    if prediction == 1:
        st.error(f"⚠️ **تنبيه بالخطر المالي:** هناك احتمالية مرتفعة لتعثر أو تأخر الطالب في السداد.")
        st.metric(label="احتمالية التعثر (Risk Probability)", value=f"{delay_probability:.1f}%")
        st.markdown("""
        **📥 التوصية الإدارية الاستباقية:**
        * التواصل السري مع ولي أمر الطالب لتسهيل الدفع.
        * إتاحة جدولة مرنة للرسوم أو توجيهه نحو صندوق دعم الطلاب بالكلية.
        """)
    else:
        st.success(f"✅ **حالة أمان مالي:** الطالب يظهر التزاماً وقدرة عالية على السداد بانتظام.")
        st.metric(label="احتمالية السداد المنتظم (Commitment Probability)", value=f"{commit_probability:.1f}%")
        st.markdown("""
        **📥 التوصية الإدارية الاستباقية:**
        * لا توجد حاجة لإجراء استباقي، حالة الطالب المالية مستقرة.
        """)

st.write("---")

# قسم أكاديمي احترافي جداً يوضح قوة النموذج للجنة المناقشة والأستاذ
with st.expander("🛠️ لوحة التحليل الأكاديمي والذكاء الاصطناعي (مخصصة للجنة المناقشة)"):
    st.write("هنا تظهر القوة العلمية الحقيقية لمشروعك أمام الدكتور والمشرفين:")
    
    # 1. دقة النموذج وحجم البيانات
    st.write(f"📊 **حجم عينة البيانات للتدريب (Synthetic Dataset):** `1000` طالب وطالبة (بناءً على خطة المشروع).")
    st.write(f"📈 **دقة التنبؤ المقياسة (Accuracy):** `{accuracy * 100:.1f}%` (باستخدام خوارزمية **Random Forest Classifier**).")
    
    st.write("---")
    
    # 2. ميزة مذهلة: أهمية المتغيرات
    st.write("🧠 **أهمية المتغيرات (Feature Importance) في اتخاذ القرار:**")
    st.write("يوضح الذكاء الاصطناعي النسبة المئوية لتأثير كل متغير أدخلته في النتيجة النهائية للتوقع:")
    
    importance_labels = {
        'family_income': 'الدخل الشهري للأسرة',
        'scholarship_ratio': 'نسبة المنحة الدراسية',
        'previous_delays': 'سجل التأخيرات السابقة'
    }
    
    for feat, imp in zip(features_list, importances):
        percent = imp * 100
        st.write(f"**{importance_labels[feat]}:** `{percent:.1f}%` تأثير على القرار.")
        st.progress(float(imp))
        
    st.write("---")
    
    # 3. عينة البيانات
    st.write("📋 **عينة من قاعدة البيانات لـ 1000 طالب بعد التدريب (أول 5 أسطر):**")
    st.dataframe(df.head(5))
