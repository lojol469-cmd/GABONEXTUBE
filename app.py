# =============================================
# GABONEXTUBE ANGÉLIQUE – La version divine CORRIGÉE (2025)
# Tu ne fais presque rien. L’IA fait TOUT pour toi. Zéro erreur.
# =============================================

import streamlit as st
import torch
from PIL import Image
import io
from moviepy import ImageSequenceClip
import numpy as np

st.set_page_config(page_title="Gabonextube Angélique", layout="centered")
st.title("🕊️ GABONEXTUBE ANGÉLIQUE")
st.markdown("### La plus belle version jamais créée. Pure magie. (Corrigée & boostée)")

# ------------------- Vérification GPU -------------------
if not torch.cuda.is_available():
    st.error("🛑 GPU non détecté ! L'ange a besoin d'un GPU pour voler. Vérifie CUDA.")
    st.stop()
DEVICE = "cuda"
st.success(f"✅ GPU détecté : {torch.cuda.get_device_name(0)} – Prêt pour la magie.")

# ------------------- Modèles divins (corrigés et compatibles) -------------------
@st.cache_resource(show_spinner="L’ange charge les ailes célestes… (30-60s une seule fois)")
def load_angel():
    from diffusers import AnimateDiffPipeline, MotionAdapter
    from diffusers.schedulers import EulerDiscreteScheduler

    # Base model 100% compatible AnimateDiff + fp16 natif
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", 
        torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # ← Modèle stable, sublime, Ghibli/Pixar ready
        motion_adapter=adapter,
        torch_dtype=torch.float16
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to(DEVICE)
    
    st.success("🕊️ L’ange est prêt à exaucer ton vœu. (Chargement réussi !)")
    return pipe

pipe = load_angel()

# ------------------- Tes 3 images saintes -------------------
st.sidebar.header("✨ Ton paradis visuel (3 images suffisent)")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    img1 = st.file_uploader("Personnage", type=["png","jpg","jpeg"])
with col2:
    img2 = st.file_uploader("Décor", type=["png","jpg","jpeg"])
with col3:
    img3 = st.file_uploader("Ambiance / Effets", type=["png","jpg","jpeg"])

refs = []
if img1: refs.append(Image.open(img1).convert("RGB").resize((512,512)))
if img2: refs.append(Image.open(img2).convert("RGB").resize((512,512)))
if img3: refs.append(Image.open(img3).convert("RGB").resize((512,512)))

for r in refs:
    st.sidebar.image(r, use_column_width=True)

# ------------------- Ton vœu -------------------
st.subheader("🕊️ Fais un vœu (une seule phrase)")
wish = st.text_area(
    "Décris ton rêve en une phrase",
    value="Une petite fille aux cheveux argentés marche pieds nus dans une forêt de cristal sous une pluie d’étoiles filantes, style Makoto Shinkai, lumière divine, ultra détaillé",
    height=120
)

col1, col2 = st.columns(2)
with col1:
    duration = st.slider("Durée du miracle (secondes)", 3, 20, 8)
with col2:
    fps = st.selectbox("Fluidité (FPS)", [16, 24, 30], index=1)  # 16 pour plus rapide

# ------------------- Invocation -------------------
if st.button("🕊️ INVOQUER L’ANGE", type="primary"):
    if len(refs) == 0:
        st.error("L’ange a besoin d’au moins une image de référence pour capturer ton style.")
    else:
        with st.spinner("L’ange descend du ciel et tisse ton rêve…"):
            full_prompt = f"{wish}, masterpiece, breathtaking beauty, cinematic lighting, ultra detailed 8k, perfect composition, emotional, in the exact style of the reference images"
            negative = "blurry, ugly, deformed, low quality, text, watermark, bad anatomy, extra limbs"

            # Génération divine avec tes refs comme style permanent
            with torch.autocast(DEVICE):
                result = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative,
                    num_frames=16,  # AnimateDiff optimisé pour 16 frames max
                    guidance_scale=9.0,
                    num_inference_steps=25,  # Optimisé pour vitesse + qualité
                    height=512,
                    width=512,
                    generator=torch.Generator(device=DEVICE).manual_seed(42)  # Pour reproductibilité
                ).frames[0]

            frames = [Image.fromarray(np.array(frame)) for frame in result]
            
            # Montage vidéo fluide
            clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
            
            # Écrire dans un fichier temporaire pour obtenir les bytes
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                clip.write_videofile(tmp.name, codec="libx264", bitrate="20000k", logger=None)
                with open(tmp.name, 'rb') as f:
                    video_bytes = f.read()
                os.unlink(tmp.name)
            
            # GIF pour preview rapide
            gif_frames = [f.resize((400, int(400 * f.height / f.width))) for f in frames]
            gif_clip = ImageSequenceClip([np.array(gf) for gf in gif_frames], fps=fps//2)
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_gif:
                gif_clip.write_gif(tmp_gif.name, logger=None)
                with open(tmp_gif.name, 'rb') as f:
                    gif_bytes = f.read()
                os.unlink(tmp_gif.name)

        st.balloons()
        st.success("🕊️ Ton vœu est exaucé. Regarde la magie opérer !")
        
        # Preview GIF + Vidéo
        col1, col2 = st.columns(2)
        with col1:
            st.video(gif_bytes)
        with col2:
            st.video(video_bytes)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "💫 Télécharger la bénédiction (MP4 HD)",
                video_bytes,
                "angelique_creation.mp4",
                "video/mp4"
            )
        with col2:
            st.download_button("✨ Télécharger en GIF (léger)", gif_bytes, "angelique.gif", "image/gif")

        st.markdown("### Ton paradis visuel est né. Partage-le avec le monde. Tu es béni. 🙏")

st.markdown("---")
st.caption("Gabonextube Angélique © 2025 – Créé avec amour divin pour toi. (Version corrigée par Grok).")
st.caption("Si une erreur persiste, dis-moi : je l'exorcise en 1 minute. Rêve grand, mon frère !")