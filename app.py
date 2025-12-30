import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from model import get_model

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)

model = get_model(pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_bone_age(image, gender):
    if image is None:
        return "Görüntü yükleyin"
    
    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    gender_val = 1.0 if gender == "Erkek" else 0.0
    gender_tensor = torch.tensor([[gender_val]]).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor, gender_tensor)
    
    months = prediction.item()
    years = months / 12
    
    return f"**{months:.0f} ay** ({years:.1f} yıl)"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦴 Kemik Yaşı Tahmini")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="El Röntgeni", height=280)
            gender_input = gr.Radio(["Erkek", "Kadın"], label="Cinsiyet", value="Erkek")
            submit_btn = gr.Button("Tahmin Et", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Sonuç")
            output = gr.Markdown("Görüntü yükleyin")
    
    submit_btn.click(predict_bone_age, [image_input, gender_input], output)


if __name__ == "__main__":
    demo.launch(share=True)
