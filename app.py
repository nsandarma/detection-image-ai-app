import streamlit as st
from preprocess import predict

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
