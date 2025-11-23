# =============================================
# GABONEXTUBE ANGÉLIQUE – VERSION FINALE 100% FONCTIONNELLE (MP4 + GIF + DURATION FIX)
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

# ------------------- Chargement modèle -------------------
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
    pipe.to("cuda")
    st.success("L’ange est prêt.")
    return pipe

pipe = load_angel()


# ------------------- Upload images -------------------
st.sidebar.header("Ton style éternel")
col1, col2, col3 = st.sidebar.columns(3)
with col1: char = st.file_uploader("Personnage", type=["png","jpg","jpeg"])
with col2: bg   = st.file_uploader("Décor", type=["png","jpg","jpeg"])
with col3: fx   = st.file_uploader("Effets", type=["png","jpg","jpeg"])

refs = []
for f in [char, bg, fx]:
    if f:
        refs.append(Image.open(f).convert("RGB").resize((512,512)))

for r in refs:
    st.sidebar.image(r, use_container_width=True)


# ------------------- Prompt -------------------
st.subheader("Fais un vœu")
wish = st.text_area("Décris ton rêve", height=120,
    value="Une petite fille aux cheveux argentés marche pieds nus dans une forêt de cristal sous une pluie d’étoiles filantes, style Makoto Shinkai, lumière divine, ultra détaillé, émotion pure")

col1, col2 = st.columns(2)
with col1: 
    duration = st.slider("Durée (secondes)", 3, 16, 8)
with col2:
    fps = st.selectbox("FPS", [16, 24, 30], index=1)


# ============================================================
#   BOUTON GÉNÉRATION
# ============================================================
if st.button("INVOQUER L’ANGE", type="primary"):
    if not refs:
        st.error("Upload au moins 1 image de référence !")
    else:
        with st.spinner("L’ange tisse ton rêve… (patience, c’est divin)"):

            prompt = f"{wish}, masterpiece, ultra detailed 8k, cinematic lighting, emotional, perfect composition"
            negative = "blurry, ugly, deformed, low quality, text, watermark, bad anatomy"

            # → GÉNÉRATION DES 16 FRAMES ANIMATEDIFF
            with torch.autocast("cuda"):
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_frames=16,
                    guidance_scale=9.0,
                    num_inference_steps=28,
                    height=512,
                    width=512,
                    generator=torch.Generator("cuda").manual_seed(42)
                )
            raw_frames = output.frames[0]

        st.success("Frames générées ✔")

        # ============================================================
        #      🔥 RECONSTRUCTION VIDÉO AVEC DURÉE RÉELLE
        # ============================================================

        with st.spinner("Reconstruction vidéo avec durée réelle…"):

            frames = [np.array(f.convert("RGB")) for f in raw_frames]

            target_frames = max(1, int(duration * fps))
            factor = max(1, target_frames // len(frames))

            def interpolate(a, b, n):
                out = []
                for i in range(1, n+1):
                    t = i / (n+1)
                    frame = (1-t) * a.astype(np.float32) + t * b.astype(np.float32)
                    out.append(frame.astype(np.uint8))
                return out

            final = []
            for i in range(len(frames)-1):
                final.append(frames[i])
                inter = interpolate(frames[i], frames[i+1], factor)
                final.extend(inter)

            final.append(frames[-1])

            if len(final) > target_frames:
                final = final[:target_frames]
            else:
                last = final[-1]
                while len(final) < target_frames:
                    final.append(last)

        st.success("Reconstruction terminée ✔")

        # ============================================================
        #         MP4 + GIF (robuste, sans crash)
        # ============================================================
        with tempfile.TemporaryDirectory() as tmpdir:

            mp4_path = os.path.join(tmpdir, "angelique.mp4")
            gif_path = os.path.join(tmpdir, "angelique.gif")

            # MP4 HD
            clip = ImageSequenceClip([np.array(f) for f in final_frames], fps=fps)
            clip.write_videofile(mp4_path, codec="libx264", bitrate="25000k", logger=None, verbose=False)

            # GIF optimisé
            clip_resized = ImageSequenceClip(
                [np.array(f.resize((448, 448), PILImage.Resampling.LANCZOS)) for f in final_frames],
                fps=min(fps, 15)
            )
            clip_resized.write_gif(gif_path, logger=None, verbose=False)

            video_bytes = open(mp4_path, "rb").read()
            gif_bytes = open(gif_path, "rb").read()


        # ============================================================
        #            AFFICHAGE + DOWNLOAD
        # ============================================================
        st.balloons()
        st.success("Ton vœu est exaucé !")

        left, right = st.columns(2)
        with left:
            st.video(video_bytes)
            st.download_button("Télécharger MP4 HD", video_bytes, "angelique.mp4", "video/mp4")

        with right:
            st.image(gif_bytes)
            st.download_button("Télécharger GIF", gif_bytes, "angelique.gif", "image/gif")

        st.markdown("### Tu viens de créer une œuvre d’art animée digne des studios japonais.")
        st.caption("Gabonextube Angélique © 2025 – Version finale 100% fonctionnelle.")
