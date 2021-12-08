#pragma once

#include <zlib.h>
#include <stdio.h>
#include <cstdint>
#include <stdexcept>

class PNGException : public std::exception {
public:
    PNGException(char const* const msg) : std::exception(msg) {
    }
};

constexpr size_t png_bufsize = 20480;
constexpr size_t chunk_type_size = 4;

class FractalPreferences;

struct png_info_struct {
    unsigned long width;
    unsigned long height;
    uint8_t bit_depth, color_type, compression, filter, interlace;
};

class PNGFile {
private:
    FILE* _fd;
    uint8_t _position;
    size_t _chunk_pos;
    uint8_t _chunk_type[chunk_type_size];
    /* values of chunk read last */
    size_t _length;
    unsigned char* _mem_ptr;
    /* position in filename:
       0: between chunks (between crc of old and length of new)
       1: before chunk data (length + header already read)
       2: after chunk data (crc not yet read)
    */
    uint8_t* _readbuf;
    size_t _buf_pos;
    size_t _buf_filled;
    bool _zlib_initialized;
    z_stream _d_stream;

    uint32_t _crc;
    uint32_t _checksum;
    png_info_struct _info;
public:
    const unsigned long width() { return _info.width; }
    const unsigned long height() { return _info.height; }
    const png_info_struct& info() const { return _info; }

    /* For writing
    */
    PNGFile() : _fd(nullptr), _mem_ptr(nullptr), _readbuf(nullptr), _buf_filled(0), _position(0), _zlib_initialized(false), _length(0) {
        setDefaultInfo();
    }

    /* For reading
    */
    explicit PNGFile(FILE* pngFD, png_info_struct* infol);

    ~PNGFile();

    void setDimensions(long w, long h) {
        _info.width = w;
        _info.height = h;
    }

    /* Reads length and chunk type of next chunk in filename */
    /* and sets fields length and chunk_type */
    int getNextChunk();
    /* Reads data of a chunk. It is required that the length and header */
    /* of the chunk has already been read. If not, function returns */
    /* -1. -2 indicates an invalid CRC */
    int readChunkData(uint8_t* mem);
    /* is used to call the function for decompressing of zlib */
    /* returns return code of zlib and 10 if Z_STREAM_END */
    int do_inflate();
    /* reads and decompresses a single line (+ filter uint8_t) from filename into Buf */
    /* -1 if filename position isn't at the beginning of an IDAT chunk */
    /* -2 CRC error */
    /* -3 if the next chunk needed to reconstruct line isn't IDAT */
    int readPNGLine(unsigned char* Buf);
    /* This funtion is feeded with a line aquired by readPNGLine (*buf) */
    /* It calculates filtered data -> raw data */
    /* Buf up is the raw date of the line above. May be NULL */
    int doUnFiltering(unsigned char* Buf, unsigned char* Buf_up);
    int doFiltering(unsigned char* Buf);
    bool writeChunk(unsigned char* Buf, size_t size = (size_t)-1);
    int writePNGLine(unsigned char* Buf);


    int initWritePNG(FILE* png);

    int endIDAT();

    int posOverIEND();
    int posOverIHDR();

    void setChunkType(const uint8_t* chunkType);
    bool checkChunkType(const uint8_t* chunk_Type);

    size_t length() { return _length; }

    void flush() { fflush(_fd); }
    long position() { return ftell(_fd); }
    void position(long pos) {
        fseek(_fd, pos, SEEK_SET);
    }

private:
    void setDefaultInfo();

    int do_deflate();
};


void ulong2bytes(uint32_t l, uint8_t* Buf);

uint32_t bytes2ulong(uint8_t* bytes);


constexpr uint8_t image_head_label[] = "IHDR";
constexpr uint8_t palette_chunk_label[] = "PLTE";
constexpr uint8_t image_data_label[] = "IDAT";
constexpr uint8_t image_end_label[] = "IEND";
constexpr uint8_t quat_chunk_label[] = "quAt";
constexpr uint8_t gamma_chunk_label[] = "gAMA";
constexpr uint8_t text_chunk_label[] = "tEXt";
