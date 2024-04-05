import streamlit as st
import os
from PIL import Image
from preprocess import predict,preprocess_image

def main():
    st.title("Deteksi Gambar AI")

    im = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if im is not None:
        out = predict(im)
        st.image(im, caption=f'Gambar yang diunggah {out}', use_column_width=True)

    else:
        st.info('Silakan unggah file gambar.')

if __name__ == "__main__":
    main()
