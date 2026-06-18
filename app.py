from flask import Flask, render_template, request
from router_model import RouterModel
import base64

app = Flask(__name__)

router = RouterModel(model_path='combined_model.pth')

# الفئات العشوائية التي دربت الموديل عليها لرفضها
REJECTED_CLASSES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

PLANT_INFO = {
    'Apple___healthy': {'ar_name': 'تفاح - سليم', 'ar_treat': 'النبات بصحة جيدة، استمر في العناية به.', 'en_treat': 'Plant is healthy, keep up the good care.'},
    'Apple___Black_rot': {'ar_name': 'تفاح - العفن الأسود', 'ar_treat': 'قم بإزالة الأجزاء المصابة واستخدم مبيد فطري نحاسي.', 'en_treat': 'Remove infected parts and use a copper fungicide.'},
    'Tomato___healthy': {'ar_name': 'طماطم - سليم', 'ar_treat': 'النبات بصحة جيدة.', 'en_treat': 'Plant is healthy.'},
    'Tomato___Late_blight': {'ar_name': 'طماطم - اللفحة المتأخرة', 'ar_treat': 'تجنب الري من الأعلى واستخدم مبيدات فطرية فوراً.', 'en_treat': 'Avoid overhead watering and apply fungicides immediately.'},
    'Potato___Early_blight': {'ar_name': 'بطاطس - اللفحة المبكرة', 'ar_treat': 'تأكد من تهوية التربة واستخدم مبيدات فطرية.', 'en_treat': 'Ensure soil aeration and use fungicides.'},
    'Corn_(maize)___healthy': {'ar_name': 'ذرة - سليم', 'ar_treat': 'النبات سليم وبصحة جيدة.', 'en_treat': 'Plant is healthy.'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="لم يتم اختيار صورة")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="اسم الملف فارغ")

        image_bytes = file.read()
        
        try:
            predicted_class = router.predict(image_bytes)
        except Exception as e:
            return render_template('index.html', error=f"حدث خطأ أثناء تحليل الصورة: {str(e)}")

        # إذا كانت الصورة تنتمي لأحد الفئات العشوائية (كلب، سيارة، إلخ) ارفضها
        if predicted_class in REJECTED_CLASSES:
            return render_template('index.html', error="هذه ليست صورة لنبات مدعوم، يرجى التأكد من تصوير أوراق النباتات المسموحة فقط.")

        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        if '___' in predicted_class:
            parts = predicted_class.split('___')
            en_plant_name = parts[0].replace('_', ' ')
            en_disease_name = parts[1].replace('_', ' ')
        else:
            en_plant_name = predicted_class
            en_disease_name = "Unknown"

        en_display_name = f"{en_plant_name} ({en_disease_name})"

        default_info = {
            'ar_name': f"{en_plant_name} - {en_disease_name}", 
            'ar_treat': 'يرجى استشارة مهندس زراعي مختص لتحديد العلاج المناسب لهذه الحالة.',
            'en_treat': 'Please consult an agricultural engineer for proper treatment.'
        }
        
        info = PLANT_INFO.get(predicted_class, default_info)

        result = {
            'image_url': image_url,
            'en': {'name': en_display_name, 'treatment': info['en_treat']},
            'ar': {'name': info['ar_name'], 'treatment': info['ar_treat']}
        }
        
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)