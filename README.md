# RAG MCP Agent

RAG MCP Agent adalah sebuah proyek asisten AI berbasis Retrieval-Augmented Generation (RAG) yang dapat menjawab pertanyaan pengguna berdasarkan dokumen pribadi atau perusahaan yang disediakan. Proyek ini memanfaatkan teknologi Large Language Model (LLM) dari OpenAI, integrasi LangChain, serta komunikasi client-server menggunakan protokol MCP (Model Context Protocol).

## Teknologi yang Digunakan

- **Python 3.10+**
- **OpenAI GPT (gpt-3.5-turbo, gpt-4o)**
- **LangChain & LangGraph**: Untuk workflow agent dan integrasi LLM
- **FAISS**: Untuk vektorisasi dan pencarian dokumen
- **dotenv**: Untuk manajemen environment variable
- **MCP (Model Context Protocol)**: Untuk komunikasi client-server
- **nest_asyncio**: Untuk mendukung event loop asyncio di lingkungan tertentu

## Struktur Folder

```
RAG-mcp-agent/
├── app/
│   ├── client.py
│   ├── main.py
│   └── server.py
├── documents/
│   ├── personal.txt
│   └── your_document.txt
├── .env
├── requirements.txt
└── README.md
```

## Cara Instalasi & Menjalankan Project

1. **Clone repository ini**

2. **Install dependensi**

    ```bash
    pip install -r requirements.txt
    ```

3. **Setup Environment Variable**

    Buat file `.env` di root project. Contoh isi yang perlu disesuaikan:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    ```

4. **Tambahkan Dokumen Pribadi**

    - Masukkan dokumen pribadi Anda (format `.txt`) ke dalam folder `documents/`.
    - Contoh: `documents/your_document.txt` berisi profil pribadi atau deskripsi perusahaan.

5. **Menjalankan Agent**

    Jalankan agent dari terminal:

    ```bash
    cd app
    python main.py
    ```

    Anda dapat langsung mengetik pertanyaan di terminal. Untuk keluar, ketik `exit` atau `quit`.

## Cara Kerja Singkat

- **Server** (`server.py`): Membaca dokumen dari folder `documents/`, melakukan indexing dengan FAISS, dan menyediakan tool `read_document` untuk menjawab pertanyaan berbasis dokumen.
- **Client** (`client.py` & `main.py`): Menghubungkan ke server, mengelola workflow percakapan, dan mengarahkan pertanyaan ke tool yang sesuai.
- **Agent**: Secara otomatis menggunakan tool `read_document` jika pertanyaan terkait dokumen pribadi/perusahaan.

## Catatan

- Pastikan API key OpenAI valid dan memiliki kuota.
- Semua dokumen yang ingin digunakan sebagai sumber jawaban harus berada di folder `documents/`.
- Untuk menambah dokumen, cukup tambahkan file `.txt` baru ke folder tersebut.

## Lisensi

Proyek ini hanya untuk keperluan pembelajaran dan pengembangan internal.
