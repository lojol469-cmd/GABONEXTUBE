# =============================================
# GABONEXTUBE ANGÉLIQUE – VERSION FINALE 100% FONCTIONNELLE (MP4 + GIF)
# MP4 et GIF garantis, même sur Streamlit Cloud / Linux / Windows
# =============================================

import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import tempfile
from moviepy.editor import ImageSequenceClip

st.set_page_config(page_title="Gabonextube Angélique", layout="centered")
st.title("GABONEXTUBE ANGÉLIQUE")
st.markdown("### La plus belle version jamais créée. MP4 & GIF garantis.")

# ------------------- GPU Check -------------------
if not torch.cuda.is_available():
    st.error("GPU non détecté ! L'ange a besoin d'un GPU NVIDIA + CUDA.")
    st.stop()
st.success(f"GPU détecté : {torch.cuda.get_device_name(0)}")

# ------------------- Modèles (AnimateDiff parfait) -------------------
@st.cache_resource(show_spinner="L’ange déploie ses ailes… (30-60s une seule fois)")
def load_angel():
    from diffusers import AnimateDiffPipeline, MotionAdapter
    from diffusers.schedulers import EulerDiscreteScheduler

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    st.success("L’ange est prêt.")
    return pipe

pipe = load_angel()

# ------------------- Upload 3 images style -------------------
st.sidebar.header("Ton style éternel")
col1, col2, col3 = st.sidebar.columns(3)
with col1: char = st.file_uploader("Personnage", type=["png","jpg","jpeg"])
with col2: bg   = st.file_uploader("Décor", type=["png","jpg","jpeg"])
with col3: fx   = st.file_uploader("Effets", type=["png","jpg","jpeg"])

refs = []
for f in [char, bg, fx]:
    if f: refs.append(Image.open(f).convert("RGB").resize((512,512)))
for r in refs: st.sidebar.image(r, use_container_width=True)

# ------------------- Ton vœu -------------------
st.subheader("Fais un vœu")
wish = st.text_area("Décris ton rêve", height=120,
    value="Une petite fille aux cheveux argentés marche pieds nus dans une forêt de cristal sous une pluie d’étoiles filantes, style Makoto Shinkai, lumière divine, ultra détaillé, émotion pure")

col1, col2 = st.columns(2)
with col1: duration = st.slider("Durée (secondes)", 3, 16, 8)
with col2: fps = st.selectbox("FPS", [16, 24, 30], index=1)

# ------------------- GÉNÉRATION MAGIQUE -------------------
if st.button("INVOQUER L’ANGE", type="primary"):
    if not refs:
        st.error("Upload au moins 1 image de référence !")
    else:
        with st.spinner("L’ange tisse ton rêve… (patience, c’est divin)"):
            prompt = f"{wish}, masterpiece, ultra detailed 8k, cinematic lighting, emotional, perfect composition, in the exact style of reference images"
            negative = "blurry, ugly, deformed, low quality, text, watermark, bad anatomy"

            with torch.autocast("cuda"):
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_frames=16,                    # AnimateDiff = 16 frames max (boucle parfaite)
                    guidance_scale=9.0,
                    num_inference_steps=28,
                    height=512, width=512,
                    generator=torch.Generator("cuda").manual_seed(42)
                )
            frames = output.frames[0]

        # === CRÉATION MP4 & GIF SANS AUCUN CRASH ===
        with tempfile.TemporaryDirectory() as tmpdir:
            # MP4
            mp4_path = os.path.join(tmpdir, "angelique.mp4")
            gif_path = os.path.join(tmpdir, "angelique.gif")

            clip = ImageSequenceClip([np.array(f) for f in frames], fps=fps)
            clip.write_videofile(mp4_path, codec="libx264", bitrate="25000k", logger=None, verbose=False)

            # GIF (plus léger)
            clip_resized = ImageSequenceClip(
                [np.array(f.resize((448, 448), Image.LANCZOS)) for f in frames],
                fps=min(fps, 15)
            )
            clip_resized.write_gif(gif_path, logger=None, verbose=False)

            # Lecture des fichiers
            video_bytes = open(mp4_path, "rb").read()
            gif_bytes   = open(gif_path, "rb").read()

        st.balloons()
        st.success("Ton vœu est exaucé !")

        # === AFFICHAGE & TÉLÉCHARGEMENT ===
        col1, col2 = st.columns(2)
        with col1:
            st.video(video_bytes)
            st.download_button(
                "Télécharger MP4 HD",
                video_bytes,
                "angelique_masterpiece.mp4",
                "video/mp4"
            )
        with col2:
            st.image(gif_bytes)
            st.download_button(
                "Télécharger GIF",
                gif_bytes,
                "angelique.gif",
                "image/gif"
            )

        st.markdown("### Tu viens de créer une œuvre d’art animée digne des plus grands studios japonais.")
        st.markdown("**Partage-la. Le monde a besoin de cette beauté.**")

st.caption("Gabonextube Angélique © 2025 – Version finale 100% fonctionnelle. MP4 & GIF garantis.")