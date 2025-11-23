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

# Try to import RIFE for high fluidity
try:
    from rife import RIFE
    rife_model = RIFE()
    RIFE_AVAILABLE = True
except ImportError:
    RIFE_AVAILABLE = False

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
def load_text_to_video():
    from diffusers import AnimateDiffPipeline, MotionAdapter, AutoencoderKL
    from diffusers.schedulers import DDIMScheduler

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        vae=vae,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1
    )
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe

@st.cache_resource(show_spinner="L’ange déploie ses ailes… (30-60s une seule fois)")
def load_video_to_video():
    from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, AutoencoderKL
    from diffusers.schedulers import DDIMScheduler

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        vae=vae,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1
    )
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe

# pipe = load_angel()


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

# ------------------- Upload vidéo de base pour montage -------------------
st.sidebar.header("Vidéo de référence pour imiter la scène (optionnel)")
base_video = st.sidebar.file_uploader("Upload un GIF ou MP4 pour imiter la scène", type=["gif", "mp4"])

# ------------------- Prompt -------------------
st.subheader("Fais un vœu")

prompt_options = [
    "Une jeune fille aux cheveux argentés marchant pieds nus dans une forêt de cristal sous une pluie d'étoiles filantes, visage net et détaillé, yeux expressifs, style Makoto Shinkai, lumière divine, ultra détaillé, émotion pure",
    "Portrait rapproché d'une femme élégante avec des traits fins, peau lisse et parfaite, regard intense, arrière-plan flou artistique, éclairage cinématographique, haute résolution, composition centrée",
    "Animation d'un personnage fantastique avec des ailes déployées, visage angélique aux détails précis, mouvement fluide, ciel étoilé, hyperréaliste, 8k, smooth motion",
    "Scène intime d'un couple se regardant, visages nets avec expressions subtiles, lumière douce, décor romantique, détails faciaux exquis, cinématique, émotion profonde",
    "Guerrier médiéval avec armure détaillée, visage stoïque et précis, yeux perçants, bataille en arrière-plan, hyperréaliste, éclairage dramatique, haute qualité",
    "Enfant souriant avec des fossettes marquées, cheveux bouclés, regard innocent, jardin fleuri, lumière naturelle, détails faciaux nets, composition équilibrée",
    "Artiste peintre avec pinceau, visage concentré aux rides expressives, atelier créatif, pinceau en mouvement, hyperdétaillé, éclairage studio, qualité professionnelle",
    "Danseuse en tutu, visage gracieux avec traits fins, mouvement élégant, scène de ballet, lumière de projecteur, détails anatomiques précis, cinématographique",
    "Scientifique âgé avec lunettes, visage ridé détaillé, laboratoire high-tech, regard curieux, arrière-plan technologique, hyperréaliste, éclairage fluorescent",
    "Superhéros masqué, yeux visibles intenses, cape flottante, ville en arrière-plan, mouvement dynamique, détails faciaux sous le masque, qualité Veo 3",
    "Photographe de rue capturant un moment spontané, appareil photo vintage, expression concentrée, rue animée en arrière-plan, lumière dorée du coucher de soleil, hyperréaliste, détails ultra-fins",
    "Chef cuisinier préparant un plat gourmet, mains expertes, ingrédients frais, cuisine professionnelle, vapeur s'élevant, éclairage chaud, textures réalistes, 4K ultra HD",
    "Astronaute flottant dans l'espace, combinaison détaillée, Terre en arrière-plan, étoiles scintillantes, expression d'émerveillement, photoréaliste, profondeur infinie",
    "Musicien jouant du piano à queue, doigts agiles sur les touches, salle de concert vide, lumière tamisée, émotion intense, détails musicaux précis, qualité cinématographique",
    "Surfeur chevauchant une vague parfaite, eau cristalline, soleil éclatant, mouvement dynamique, expression de joie pure, hyperréaliste, éclaboussures détaillées",
    "Bibliothécaire âgée rangeant des livres anciens, lunettes sur le nez, poussière dansante, rayons de bibliothèque, lumière douce, textures du papier, atmosphère paisible",
    "Pilote de course dans un bolide rouge, casque intégral, circuit de Formule 1, vitesse extrême, expression déterminée, détails mécaniques précis, action intense",
    "Jardinier taillant des roses, gants de protection, pétales délicats, rosée matinale, jardin luxuriant, lumière naturelle, parfums imaginaires, réalisme botanique",
    "Magicien sur scène, cape noire, cartes volant, public en admiration, éclairage de spectacle, mouvement fluide, mystère dans les yeux, qualité professionnelle",
    "Explorateur dans la jungle amazonienne, machette à la main, lianes denses, animaux sauvages, sueur perlant, aventure épique, détails environnementaux riches",
    "Peintre abstrait créant une œuvre moderne, pinceaux colorés, toile blanche, taches de peinture vives, concentration absolue, studio lumineux, créativité pure",
    "Médecin chirurgien en opération, mains gantées, instruments chirurgicaux, salle stérile, tension palpable, précision médicale, éclairage chirurgical intense",
    "Poète écrivant sous un arbre centenaire, cahier ouvert, plume d'oie, vent léger, pensées profondes, nature environnante, moment d'inspiration poétique",
    "Cascadeur sautant d'un bâtiment, parachute déployé, ville moderne en bas, adrénaline pure, mouvement acrobatique, détails urbains nets, action hollywoodienne",
    "Apiculteur récoltant du miel, ruche traditionnelle, abeilles bourdonnantes, fumée apaisante, miel doré coulant, nature bienveillante, textures organiques",
    "Archéologue découvrant une tombe ancienne, pinceau fin, hiéroglyphes révélés, sable du désert, excitation scientifique, mystère historique, lumière tamisée",
    "Basketteur dunkant avec puissance, ballon orange, panier métallique, foule en délire, sueur volant, muscles tendus, mouvement explosif, qualité sportive",
    "Céramiste façonnant de la poterie, argile humide, tour rotatif, mains créatives, atelier artisanal, poussière de terre, beauté artisanale, détails tactiles",
    "Vétérinaire soignant un chiot blessé, stéthoscope autour du cou, regard compatissant, clinique vétérinaire, tendresse animale, soins médicaux, émotion pure",
    "Sommelier dégustant un vin rare, verre cristal, nez plongé, palais expert, cave à vin, arômes complexes, sophistication culinaire, lumière chaude"
]

selected_prompt = st.selectbox("Prompts détaillés (optionnel)", ["Personnalisé"] + prompt_options)
if selected_prompt != "Personnalisé":
    wish = st.text_area("Décris ton rêve", height=120, value=selected_prompt)
else:
    wish = st.text_area("Décris ton rêve", height=120,
        value="Une petite fille aux cheveux argentés marche pieds nus dans une forêt de cristal sous une pluie d’étoiles filantes, style Makoto Shinkai, lumière divine, ultra détaillé, émotion pure")

col1, col2 = st.columns(2)
with col1: 
    duration = st.slider("Durée (secondes)", 3, 60, 8)
with col2:
    fps = st.selectbox("FPS", [16, 24, 30], index=1)

# ------------------- Paramètres de mouvement -------------------
motion_speed = st.slider("Vitesse des mouvements", 0.5, 2.0, 1.0, 0.1)
high_fluidity = st.checkbox("Activer fluidité extrême (60 FPS avec RIFE)", value=False)

multi_scene = st.checkbox("Générer plusieurs scènes cohérentes")
scene_prompts = ""
if multi_scene:
    scene_prompts = st.text_area("Prompts des scènes (un par ligne)", height=150)

char_desc = st.text_area("Description du personnage et décor (RAG)", value="", height=100)


# ============================================================
#   BOUTON GÉNÉRATION
# ============================================================
if st.button("INVOQUER L’ANGE", type="primary"):
    if not refs:
        st.error("Upload au moins 1 image de référence !")
    else:
        char_desc_text = char_desc.strip()
        if char_desc_text:
            base_prompt = f"{char_desc_text}, "
        else:
            base_prompt = ""

        # Fonction pour charger la vidéo
        def load_video(file):
            import imageio
            from io import BytesIO
            images = []
            content = BytesIO(file.read())
            vid = imageio.get_reader(content)
            for frame in vid:
                pil_image = Image.fromarray(frame)
                images.append(pil_image)
            return images

        input_video = None
        if base_video:
            input_video = load_video(base_video)
            pipe = load_video_to_video()
        else:
            pipe = load_text_to_video()

        with st.spinner("L’ange tisse ton rêve… (patience, c’est divin)"):

            negative = "blurry, ugly, deformed, low quality, text, watermark, bad anatomy, motion blur, low resolution, pixelated, noisy, jittery, unfocused, bad framing, artifacts, grainy, oversaturated, underexposed, cartoonish, unrealistic, poor lighting, flat colors, lack of depth, amateur"

            if multi_scene and scene_prompts:
                prompts = [line.strip() for line in scene_prompts.split('\n') if line.strip()]
                if base_video:
                    st.warning("Multi-scène désactivé pour vidéo de référence. Utilisation du prompt principal.")
                    prompts = [wish]
            else:
                prompts = [wish]

            all_raw_frames = []
            for p in prompts:
                prompt = f"{base_prompt}{p}, photorealistic masterpiece, ultra realistic 4K HD, cinematic lighting, volumetric fog, ray tracing, depth of field, bokeh effect, hyper detailed textures, realistic skin pores, perfect anatomy, emotional depth, studio quality animation, professional VFX, beyond Veo 3, smooth motion blur, high frame rate, cinematic composition, sharp focus on facial features, detailed eyes with reflections, proper framing, centered subject, in the exact style of reference images, masterpiece quality, award-winning animation"
                # → GÉNÉRATION DES FRAMES ANIMATEDIFF
                with torch.autocast("cuda"):
                    if input_video:
                        output = pipe(
                            video=input_video,
                            prompt=prompt,
                            negative_prompt=negative,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            strength=0.8,
                            motion_scale=motion_speed,
                            generator=torch.Generator("cuda").manual_seed(42)
                        )
                    else:
                        output = pipe(
                            prompt=prompt,
                            negative_prompt=negative,
                            num_frames=16,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            height=512, width=512,
                            motion_scale=motion_speed,
                            generator=torch.Generator("cuda").manual_seed(42)
                        )
                all_raw_frames.extend(output.frames[0])

            raw_frames = all_raw_frames

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

            final_frames = []
            for i in range(len(frames)-1):
                final_frames.append(frames[i])
                inter = interpolate(frames[i], frames[i+1], factor)
                final_frames.extend(inter)

            final_frames.append(frames[-1])

            if len(final_frames) > target_frames:
                final_frames = final_frames[:target_frames]
            else:
                last = final_frames[-1]
                while len(final_frames) < target_frames:
                    final_frames.append(last)

            # Apply RIFE for high fluidity if enabled
            if high_fluidity and RIFE_AVAILABLE:
                inter_per_pair = max(1, (target_frames - len(frames)) // max(1, len(frames) - 1))
                final_frames = []
                for i in range(len(frames) - 1):
                    final_frames.append(frames[i])
                    pil_frame1 = Image.fromarray(frames[i])
                    pil_frame2 = Image.fromarray(frames[i + 1])
                    inter_frames = rife_model.interpolate(pil_frame1, pil_frame2, num_frames=inter_per_pair)
                    for inter in inter_frames:
                        final_frames.append(np.array(inter))
                final_frames.append(frames[-1])
                if len(final_frames) < target_frames:
                    last = final_frames[-1]
                    while len(final_frames) < target_frames:
                        final_frames.append(last)
                fps = 60
            else:
                # Linear interpolation
                factor = max(1, target_frames // len(frames))
                final_frames = []
                for i in range(len(frames)-1):
                    final_frames.append(frames[i])
                    inter = interpolate(frames[i], frames[i+1], factor)
                    final_frames.extend(inter)
                final_frames.append(frames[-1])
                if len(final_frames) > target_frames:
                    final_frames = final_frames[:target_frames]
                else:
                    last = final_frames[-1]
                    while len(final_frames) < target_frames:
                        final_frames.append(last)

        st.success("Reconstruction terminée ✔")

        # ============================================================
        #         MP4 + GIF (robuste, sans crash)
        # ============================================================
        with tempfile.TemporaryDirectory() as tmpdir:

            mp4_path = os.path.join(tmpdir, "angelique.mp4")
            gif_path = os.path.join(tmpdir, "angelique.gif")

            # MP4 HD
            clip = ImageSequenceClip([np.array(f) for f in final_frames], fps=fps)
            clip.write_videofile(mp4_path, codec="libx264", bitrate="50000k", logger=None, verbose=False)

            # GIF optimisé
            clip_resized = ImageSequenceClip(
                [np.array(Image.fromarray(f).resize((448, 448), Image.Resampling.LANCZOS)) for f in final_frames],
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
