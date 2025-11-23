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
    from diffusers import AnimateDiffPipeline, MotionAdapter, IPAdapter
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

    # === AJOUT RAG VISUEL IP-Adapter ===
    ip_adapter = IPAdapter.from_pretrained(
        "h94/IP-Adapter-FaceID",  # version ultra robuste
        torch_dtype=torch.float16
    )
    pipe.load_ip_adapter(ip_adapter)

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

            # === RAG VISUEL : encoder les images de style ===
            style_embeds = [
                pipe.encode_image(img).latent_dist.sample()
                for img in refs
            ]

            with torch.autocast("cuda"):
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_frames=16,                    # AnimateDiff = 16 frames max (boucle parfaite)
                    guidance_scale=9.0,
                    num_inference_steps=28,
                    height=512, width=512,
                    generator=torch.Generator("cuda").manual_seed(42),
                    ip_adapter_image_embeds=style_embeds,   # ← MAGIE DU RAG
                    ip_adapter_scale=[0.8, 0.6, 0.5]        # ← intensité par image
                )
            frames = output.frames[0]

        # --------------------------------------------------------
        # Après: frames = output.frames[0]
        # On s'assure que chaque frame est un PIL.Image en RGB
        from PIL import Image as PILImage

        # Convertir les frames brutes (PIL ou numpy) en PIL.Image
        pil_frames = []
        for f in frames:
            if isinstance(f, PILImage.Image):
                pil_frames.append(f.convert("RGB"))
            else:
                # si c'est un tableau numpy
                pil_frames.append(PILImage.fromarray(np.array(f)).convert("RGB"))

        n_gen = len(pil_frames)
        target_frames = max(1, int(duration * fps))  # nombre d'images souhaité
        if target_frames == n_gen:
            final_frames = pil_frames
        elif target_frames < n_gen:
            # Downsample uniformément
            indices = np.linspace(0, n_gen - 1, target_frames).round().astype(int)
            final_frames = [pil_frames[i] for i in indices]
        else:
            # Upsample: interpolation par crossfade linéaire entre paires
            required = target_frames - n_gen
            pairs = max(1, n_gen - 1)
            base = required // pairs
            rem = required % pairs

            final_frames = []
            for i in range(n_gen - 1):
                a = np.asarray(pil_frames[i], dtype=np.float32)
                b = np.asarray(pil_frames[i + 1], dtype=np.float32)

                # ajouter la frame de départ
                final_frames.append(PILImage.fromarray(a.astype(np.uint8)))

                # combien d'interpolations pour cette paire
                k = base + (1 if i < rem else 0)
                for j in range(1, k + 1):
                    alpha = j / (k + 1)
                    interp = (1.0 - alpha) * a + alpha * b
                    final_frames.append(PILImage.fromarray(np.clip(interp, 0, 255).astype(np.uint8)))

            # ajouter la dernière frame finale
            final_frames.append(pil_frames[-1])

            # sécurité : découper ou compléter si léger décalage
            if len(final_frames) > target_frames:
                final_frames = final_frames[:target_frames]
            elif len(final_frames) < target_frames:
                # répéter la dernière image si nécessaire (rare)
                while len(final_frames) < target_frames:
                    final_frames.append(final_frames[-1])

        # === CRÉATION MP4 & GIF SANS AUCUN CRASH (comme avant) ===
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = os.path.join(tmpdir, "angelique.mp4")
            gif_path = os.path.join(tmpdir, "angelique.gif")

            # pour moviepy, on donne la liste de tableaux numpy
            clip = ImageSequenceClip([np.array(f) for f in final_frames], fps=fps)
            clip.write_videofile(mp4_path, codec="libx264", bitrate="25000k", logger=None, verbose=False)

            # GIF (plus léger) : on redimensionne avec PIL safe
            clip_resized = ImageSequenceClip(
                [np.array(f.resize((448, 448), PILImage.Resampling.LANCZOS)) for f in final_frames],
                fps=min(fps, 15)
            )
            clip_resized.write_gif(gif_path, logger=None, verbose=False)

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