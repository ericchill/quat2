#pragma once

#include <zlib.h>
#include <stdio.h>
#include <cstdint>


constexpr size_t png_bufsize = 20480;
constexpr size_t chunk_type_size = 4;

struct png_internal_struct
{
   unsigned long width;
   unsigned long height;
   uint8_t bit_depth, color_type, comression, filter, interlace;
   uint8_t PLTE;
   FILE *png;
   uint8_t position;
   /* position in file:
      0: between chunks (between crc of old and length of new)
      1: before chunk data (length + header already read)
      2: after chunk data (crc not yet read)
   */
   size_t length;
   uint32_t crc;
   uint8_t chunk_type[chunk_type_size];
   /* values of chunk read last */
   unsigned char *mem_ptr;
   z_stream d_stream;
   /* for zlib library status */
   bool zlib_initialized;
   bool readbuf_initialized;
   uint8_t *readbuf;
   size_t buf_pos;
   size_t buf_filled;
   size_t chunk_pos;
   uint32_t checksum;
} ;

struct png_info_struct
{
   unsigned long width;
   unsigned long height;
   uint8_t bit_depth, color_type, compression, filter, interlace;
};

int InitPNG(FILE *png, png_info_struct *info, png_internal_struct *internal);
   /* Has to be called after opening and before reading the PNG file      */
   /* returns -1 if png signature check failed or first chunk is not IHDR */
int GetNextChunk(png_internal_struct *i);
   /* Reads length and chunk type of next chunk in file */
   /* and sets fields length and chunk_type */
int ReadChunkData(png_internal_struct *i, uint8_t *mem);
   /* Reads data of a chunk. It is required that the length and header */
   /* of the chunk has already been read. If not, function returns */
   /* -1. -2 indicates an invalid CRC */
int do_inflate(png_internal_struct *i);
   /* is used to call the function for decompressing of zlib */
   /* returns return code of zlib and 10 if Z_STREAM_END */
int ReadPNGLine(png_internal_struct *i, unsigned char *Buf);
   /* reads and decompresses a single line (+ filter uint8_t) from file into Buf */
   /* -1 if file position isn't at the beginning of an IDAT chunk */
   /* -2 CRC error */
   /* -3 if the next chunk needed to reconstruct line isn't IDAT */
int DoUnFiltering(png_internal_struct *i, unsigned char *Buf, unsigned char *Buf_up);
   /* This funtion is feeded with a line aquired by ReadPNGLine (*buf) */
   /* It calculates filtered data -> raw data */
   /* Buf up is the raw date of the line above. May be NULL */
int DoFiltering(png_internal_struct *i, unsigned char *Buf);
int InitWritePNG(FILE *png, png_info_struct *info, png_internal_struct *i);
int WriteChunk(png_internal_struct *i, unsigned char *Buf);
int WritePNGLine(png_internal_struct *i, unsigned char *Buf);
int EndIDAT(png_internal_struct *i);
int PosOverIEND(png_internal_struct *i);
int PosOverIHDR(png_internal_struct *i);
int EndPNG(png_internal_struct *i);

void setChunkType(png_internal_struct* i, const uint8_t* chunkType);
bool checkChunkType(png_internal_struct* i, const uint8_t* chunk_Type);

void ulong2bytes(uint32_t l, uint8_t* Buf);

uint32_t bytes2ulong(uint8_t* bytes);


constexpr uint8_t image_head_label[] = "IHDR";
constexpr uint8_t palette_chunk_label[] = "PLTE";
constexpr uint8_t image_data_label[] = "IDAT";
constexpr uint8_t image_end_label[] = "IEND";
constexpr uint8_t quat_chunk_label[] = "quAt";
constexpr uint8_t gamma_chunk_label[] = "gAMA";
constexpr uint8_t text_chunk_label[] = "tEXt";
