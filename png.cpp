/* Quat - A 3D fractal generation program */
/* Copyright (C) 1997,98 Dirk Meyer */
/* (email: dirk.meyer@studserv.uni-stuttgart.de) */
/* mail:  Dirk Meyer */
/*        Marbacher Weg 29 */
/*        D-71334 Waiblingen */
/*        Germany */
/* */
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */
/* */
/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */
/* */
/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>   /* malloc, free */
#include <string.h>
#include <math.h>
#include "png.h"



int do_deflate(png_internal_struct* i);
uint8_t PaethPredictor(uint8_t a, uint8_t b, uint8_t c);

/* NOTE: In this (new) version of PNG.C the buffer "readbuf" in */
/* png_internal_struct is allocated via malloc */
/* a call of EndPNG() is needed after use of these structure */
/* to deallocate the buffer */

uint8_t png_signature[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };

static uint32_t crc_table[256];
static bool crc_table_computed = false;
constexpr uint32_t crc32_coeffs = 0xedb88320L;

void make_crc_table(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            if (c & 1) {
                c = crc32_coeffs ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crc_table[n] = c;
    }
    crc_table_computed = 1;
}

uint32_t update_crc(uint32_t crc, const uint8_t* buf, size_t len) {
    unsigned long newCRC = crc;

    if (!crc_table_computed) {
        make_crc_table();
    }
    for (int n = 0; n < len; n++) {
        newCRC = crc_table[(int)((newCRC ^ (long)buf[n]) & 0xffL)] ^ (newCRC >> 8);
    }
    return newCRC;
}

uint32_t crc(uint8_t* buf, size_t len) {
    return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL;
}


void ulong2bytes(uint32_t l, uint8_t* Buf) {
    Buf[0] = static_cast<uint8_t>(l >> 24 & 0xffL);
    Buf[1] = static_cast<uint8_t>(l >> 16 & 0xffL);
    Buf[2] = static_cast<uint8_t>(l >> 8 & 0xffL);
    Buf[3] = static_cast<uint8_t>(l & 0xffL);
}

uint32_t bytes2ulong(uint8_t* bytes) {
    uint32_t l;
    l = bytes[0];
    l <<= 8;
    l |= bytes[1];
    l <<= 8;
    l |= bytes[2];
    l <<= 8;
    l |= bytes[3];
    return l;
}

uint32_t readLong(FILE* fd) {
    uint8_t longBuf[4];
    fread_s(longBuf, sizeof(longBuf), sizeof(longBuf), 1, fd);
    return bytes2ulong(longBuf);
}

int InitPNG(FILE* png,
    png_info_struct* info,
    png_internal_struct* internal) {
    unsigned char sig[sizeof(png_signature)], infoBuf[100];

    internal->readbuf_initialized = 0;     /* for EndPNG, if no valid file */
    internal->png = png;
    fseek(internal->png, 0L, SEEK_SET);
    fread(sig, 1, sizeof(sig), internal->png);
    if (0 != memcmp(sig, png_signature, sizeof(sig))) {
        return -1;
    }
    /*   if (strncmp(sig, png_signature, 8)!=0) return(-1); */
    internal->length = readLong(internal->png);
    fread(internal->chunk_type, 1, sizeof(internal->chunk_type), internal->png);
    if (0 != memcmp(internal->chunk_type, image_head_label, sizeof(internal->chunk_type))) {
        return -1;
    }
    size_t png_info_length = std::max(internal->length, sizeof(png_info_struct));
    fread_s(infoBuf, sizeof(infoBuf), 1, png_info_length, internal->png);
    info->width = bytes2ulong(&infoBuf[0]);
    info->height = bytes2ulong(&infoBuf[4]);
    info->bit_depth = infoBuf[8];
    info->color_type = infoBuf[9];
    info->compression = infoBuf[10];
    info->filter = infoBuf[11];
    info->interlace = infoBuf[12];
    internal->crc = readLong(internal->png);
    memcpy(internal, info, sizeof(png_info_struct));
    internal->PLTE = 0;
    internal->position = 0;
    internal->zlib_initialized = false;
    internal->readbuf_initialized = false;
    internal->buf_filled = 0;
    internal->chunk_pos = 0;
    internal->readbuf = NULL;
    return 0;
}

int GetNextChunk(png_internal_struct* i) {
    if (i->position == 1) {
        fseek(i->png, (long)i->length + 4, SEEK_CUR);
    } else if (i->position == 2) {
        fseek(i->png, 4L, SEEK_CUR);
    }
    i->position = 1;
    i->length = readLong(i->png);
    fread_s(i->chunk_type, sizeof(i->chunk_type), 1, sizeof(i->chunk_type), i->png);
    if (checkChunkType(i, palette_chunk_label)) {
        i->PLTE++;
    }
    return 0;
}

int ReadChunkData(png_internal_struct* i, uint8_t* mem) {
    unsigned long checksum;

    if (i->position != 1) {
        return -1;
    }
    i->mem_ptr = mem;
    fread(i->mem_ptr, 1, i->length, i->png);
    i->crc = readLong(i->png);
    checksum = update_crc(0xffffffffL, (unsigned char*)i->chunk_type, sizeof(i->chunk_type));
    checksum = update_crc(checksum, i->mem_ptr, i->length);
    checksum ^= 0xffffffffL;
    if (checksum != i->crc) {
        return -2;
    }
    i->position = 0;

    return 0;
}

int do_inflate(struct png_internal_struct* i) {
    int err;

    if (!i->zlib_initialized) {
        i->zlib_initialized = true;
        i->d_stream.zalloc = (alloc_func)0;
        i->d_stream.zfree = (free_func)0;

        err = inflateInit(&i->d_stream);
    }
    err = inflate(&i->d_stream, Z_NO_FLUSH);

    if (err == Z_STREAM_END) {
        err = inflateEnd(&i->d_stream);
        return 10;
    }
    return err;
}

int ReadPNGLine(png_internal_struct* i, unsigned char* Buf) {
    size_t toread;
    int err;
    int ByPP;

    /* Test, if chunk is IDAT */
    if (!checkChunkType(i, image_data_label)) {
        return -1;
    }

    /* Initialize on first call */
    if (i->chunk_pos == 0) {
        i->checksum = update_crc(0xffffffffL, i->chunk_type, 4);
    }
    if (!i->readbuf_initialized) {
        i->readbuf = new uint8_t[png_bufsize];
        i->readbuf_initialized = true;
    }

    if (0 == i->buf_filled) {
        toread = std::max(static_cast<size_t>(i->length - i->chunk_pos), png_bufsize);
        i->buf_filled = fread(i->readbuf, 1, toread, i->png);
        if (0 == i->buf_filled) {
            return -1;
        }
        i->buf_pos = 0;
        i->chunk_pos += i->buf_filled;
        i->checksum = update_crc(i->checksum, i->readbuf, toread);
    }

    ByPP = (int)ceil((float)i->bit_depth / 8);
    if (i->color_type == 2) {
        ByPP *= 3;
    }

    /* Only decompress a single line & filter uint8_t */
    i->d_stream.avail_out = (int)(i->width * ByPP + 1);
    i->d_stream.next_out = Buf;

    if (i->color_type == 3) {
        switch (i->bit_depth) {
        case 1: i->d_stream.avail_out = (int)(i->width / 8 + 1); break;
        case 2: i->d_stream.avail_out = (int)(i->width / 4 + 1); break;
        case 4: i->d_stream.avail_out = (int)(i->width / 2 + 1); break;
        }
    }

    do {
        i->d_stream.next_in = i->readbuf + i->buf_pos;
        i->d_stream.avail_in = static_cast<uInt>(i->buf_filled - i->buf_pos);
        err = do_inflate(i);
        i->buf_pos = i->d_stream.next_in - i->readbuf;

        /* ready ? */
        if (i->d_stream.avail_out == 0) {
            break;
        }
        /* no process made */
        if (i->d_stream.avail_out != 0 || i->buf_pos == i->buf_filled) {
            /* because chunk at end */
            if (i->chunk_pos == i->length) {
                /* verify crc */
                i->checksum ^= 0xffffffffL;
                i->crc = readLong(i->png);
                i->position = 0;
                if (i->checksum != i->crc) {
                    return -2;
                }
                /* read next chunk header */
                i->length = readLong(i->png);
                fread_s(i->chunk_type, sizeof(i->chunk_type), sizeof(i->chunk_type), 1, i->png);
                i->position = 1;
                /* is it an IDAT chunk? */
                if (0 != memcmp(i->chunk_type, image_data_label, sizeof(i->chunk_type))) {
                    return -3;
                }
                /* initialize checksum */
                i->checksum = update_crc(0xffffffffL, i->chunk_type, sizeof(i->chunk_type));
                /* prepare reading chunk data */
                i->chunk_pos = 0;
                toread = std::max(i->length, png_bufsize);
                i->buf_filled = fread_s(i->readbuf, png_bufsize, 1, toread, i->png);
                if (0 == i->buf_filled) {
                    return -1;
                }
                i->buf_pos = 0;
                i->chunk_pos += i->buf_filled;
                i->checksum = update_crc(i->checksum, i->readbuf, toread);
            } else {
                /* because 2K buffer at end */
                toread = std::max(static_cast<size_t>(i->length - i->chunk_pos), png_bufsize);
                i->buf_filled = fread_s(i->readbuf, png_bufsize, 1, toread, i->png);
                if (0 == i->buf_filled) {
                    return -1;
                }
                i->buf_pos = 0;
                i->chunk_pos += i->buf_filled;
                i->checksum = update_crc(i->checksum, i->readbuf, toread);
            }
        }
    } while (i->d_stream.avail_out != 0);

    return 0;
}

uint8_t PaethPredictor(long a, long b, long c) {
    int p;
    unsigned long pa, pb, pc;

    /* a = left, b = above, c = upper left  */
    p = a + b - c;
    pa = abs(p - a);
    pb = abs(p - b);
    pc = abs(p - c);
    if (pa <= pb && pa <= pc) {
        return static_cast<uint8_t>(a);
    } else if (pb <= pc) {
        return static_cast<uint8_t>(b);
    } else {
        return static_cast<uint8_t>(c);
    }
}

int DoUnFiltering(png_internal_struct* i, uint8_t* Buf, uint8_t* Buf_up) {
    unsigned long ByPP;
    long up, prior, upperleft;

    ByPP = (unsigned long)ceil((float)i->bit_depth / 8.0);
    if (i->color_type == 2) {
        ByPP *= 3;
    }
    switch (Buf[0]) {
    case 0: break;       /* No filter */
    case 1:              /* Sub */
        for (size_t uj = ByPP + 1; uj <= i->width * ByPP; uj++) {
            Buf[uj] = (Buf[uj] + Buf[uj - ByPP]) % 256;
        }
        Buf[0] = 0;
        break;
    case 2:              /* Up */
        if (Buf_up == NULL) {
            Buf[0] = 0;
            return 0; /* according to png spec. it is assumed that */
                       /* Buf_up is zero everywhere */
        }
        for (size_t uj = 1; uj <= i->width * ByPP; uj++) {
            Buf[uj] = (Buf[uj] + Buf_up[uj]) % 256;
        }
        Buf[0] = 0;
        break;
    case 3:             /* Average */
        for (size_t uj = 1; uj <= i->width * ByPP; uj++) {
            if (uj > ByPP) {
                prior = Buf[uj - ByPP];
            } else {
                prior = 0;
            }
            if (Buf_up != NULL) {
                up = Buf_up[uj];
            } else {
                up = 0;
            }
            Buf[uj] = (Buf[uj] + ((prior + up) >> 1)) % 256;
        }
        Buf[0] = 0;
        break;
    case 4:             /* Paeth */
        for (size_t uj = 1; uj <= i->width * ByPP; uj++) {
            if (Buf_up != NULL) {
                up = Buf_up[uj];
            } else {
                up = 0;
            }
            if (uj > ByPP) {
                prior = Buf[uj - ByPP];
            } else {
                prior = 0;
            }
            if (uj > ByPP && Buf_up != NULL) {
                upperleft = Buf_up[uj - ByPP];
            } else {
                upperleft = 0;
            }
            Buf[uj] = static_cast<uint8_t>((Buf[uj] + PaethPredictor(prior, up, upperleft)) % 256);
        }
        Buf[0] = 0;
        break;
    }
    return 0;
}

int DoFiltering(struct png_internal_struct* i, unsigned char* Buf) {
    int ByPP, j, t;

    if (Buf[0] == 0) {
        return 0; /* No filter */
    }
    if (Buf[0] == 1) {
        /* Sub       */
        ByPP = (int)ceil((double)i->bit_depth / 8);
        if (i->color_type == 2) {
            ByPP *= 3;
        }
        for (j = (int)i->width * ByPP; j >= ByPP + 1; j--) {
            t = ((int)Buf[j] - (int)Buf[j - ByPP]);
            if (t < 0) t += 256;
            Buf[j] = t % 256;
        }
    }
    return 0;
}

int InitWritePNG(FILE* png, struct png_info_struct* info, struct png_internal_struct* i) {
    unsigned long length, crc_v;
    unsigned char Buf[20];

    i->png = png;
    if (fwrite(png_signature, 1, 8, png) != 8) {
        return -1;
    }
    i->length = length = 13;
    ulong2bytes(length, Buf);
    if (fwrite(Buf, 1, 4, png) != 4) {
        return -1;
    }
    if (fwrite(image_head_label, 1, 4, png) != 4) {
        return -1;
    }
    ulong2bytes(info->width, Buf);
    ulong2bytes(info->height, Buf + 4);
    Buf[8] = info->bit_depth;
    Buf[9] = info->color_type;
    Buf[10] = info->compression;
    Buf[11] = info->filter;
    Buf[12] = info->interlace;
    if (fwrite(Buf, 1, 13, png) != 13) {
        return -1;
    }
    crc_v = update_crc(0xffffffffL, (unsigned char*)image_head_label, 4);
    crc_v = update_crc(crc_v, Buf, (long)i->length);
    crc_v ^= 0xffffffffL;
    ulong2bytes(crc_v, Buf);
    if (fwrite(Buf, 1, 4, png) != 4) {
        return -1;
    }
    memcpy(i, info, sizeof(png_info_struct));
    memcpy(i->chunk_type, image_head_label, sizeof(i->chunk_type));
    i->PLTE = 0;
    i->position = 0;
    i->zlib_initialized = 0;
    i->readbuf_initialized = 0;
    i->buf_filled = 0;
    i->chunk_pos = 0;
    i->readbuf = NULL;

    return 0;
}

int WriteChunk(png_internal_struct* i, unsigned char* buf) {
    uint32_t crc_v;
    uint8_t longBuf[sizeof(uint32_t)];

    ulong2bytes(static_cast<uint32_t>(i->length), longBuf);
    size_t written = fwrite(longBuf, sizeof(longBuf), 1, i->png);
    if (written != 1) {
        return -1;
    }
    if (fwrite(i->chunk_type, 1, sizeof(i->chunk_type), i->png) != sizeof(i->chunk_type)) {
        return -1;
    }
    if (fwrite(buf, 1, i->length, i->png) != i->length) {
        return -1;
    }
    crc_v = update_crc(0xffffffffL, const_cast<const uint8_t*>(i->chunk_type), sizeof(i->chunk_type));
    crc_v = update_crc(crc_v, buf, i->length);
    crc_v ^= 0xffffffffL;
    ulong2bytes(crc_v, longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, i->png) != 1) {
        return -1;
    }
    i->position = 0;

    return 0;
}

int do_deflate(png_internal_struct* i) {
    int err;

    if (i->zlib_initialized == 0) {
        i->zlib_initialized = 1;
        i->d_stream.zalloc = (alloc_func)Z_NULL;
        i->d_stream.zfree = (free_func)Z_NULL;
        err = deflateInit(&i->d_stream, Z_DEFAULT_COMPRESSION);
    }

    err = deflate(&i->d_stream, Z_NO_FLUSH);

    return(err);
}

int WritePNGLine(png_internal_struct* i, uint8_t* Buf) {
    int err;
    int ByPP;

    /* Initialize on first call */
    if (i->chunk_pos == 0) {
        i->checksum = update_crc(0xffffffffL, image_data_label, sizeof(i->chunk_type));
        if (fwrite(i, 1, 4, i->png) != 4) {
            return -1;   /* length as dummy */
        }
        if (fwrite("IDAT", 1, 4, i->png) != 4) {
            return -1;
        }
    }
    if (i->readbuf_initialized == 0) {
        i->readbuf = new uint8_t[png_bufsize];
        if (!i->readbuf) return(-2); /* Memory error */
        i->readbuf_initialized = 1;
    }

    ByPP = (int)ceil((double)i->bit_depth / 8);
    if (i->color_type == 2) {
        ByPP *= 3;
    }

    i->d_stream.avail_in = (int)((i->width) * ByPP + 1); /* Only compress a single line */
    i->d_stream.next_in = Buf;

    if (i->color_type == 3) {
        switch (i->bit_depth) {
        case 1: i->d_stream.avail_in = (int)((i->width) / 8 + 1); break;
        case 2: i->d_stream.avail_in = (int)((i->width) / 4 + 1); break;
        case 4: i->d_stream.avail_in = (int)((i->width) / 2 + 1); break;
        }
    }

    do {
        i->d_stream.next_out = i->readbuf;
        i->d_stream.avail_out = png_bufsize;
        err = do_deflate(i);

        i->buf_pos = (int)(i->d_stream.next_out - i->readbuf);
        i->checksum = update_crc(i->checksum, i->readbuf, i->buf_pos);
        if (fwrite(i->readbuf, 1, i->buf_pos, i->png) != i->buf_pos) {
            return -1;
        }
        i->chunk_pos += i->buf_pos;
    } while (i->d_stream.avail_out == 0);

    return 0;
}

int EndIDAT(png_internal_struct* i) {
    int err;
    uint8_t longBuf[sizeof(uint32_t)];

    if (NULL == i->readbuf) {
        return -1;  /* EndIDAT called too early */
    }
    do {
        i->d_stream.next_out = i->readbuf;
        i->d_stream.avail_out = png_bufsize;
        err = deflate(&i->d_stream, Z_FINISH);
        i->buf_pos = (int)(i->d_stream.next_out - i->readbuf);
        if (fwrite(i->readbuf, 1, i->buf_pos, i->png) != i->buf_pos) {
            return -1;
        }
        i->chunk_pos += i->buf_pos;
        i->checksum = update_crc(i->checksum, i->readbuf, i->buf_pos);
    } while (i->d_stream.avail_out == 0);
    err = deflateEnd(&i->d_stream);

    i->checksum ^= 0xffffffffL;
    ulong2bytes(i->checksum, longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, i->png) != 1) {
        return -1;
    }
    fseek(i->png, -(long)(i->chunk_pos + 12), SEEK_CUR);
    ulong2bytes(static_cast<uint32_t>(i->chunk_pos), longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, i->png) != 1) {
        return -1;
    }
    fseek(i->png, static_cast<unsigned long>(i->chunk_pos + 8), SEEK_CUR);
    i->position = 0;

    return 0;
}

int PosOverIEND(png_internal_struct* i) {
    png_internal_struct last;

    while (0 != memcmp(i->chunk_type, image_end_label, sizeof(i->chunk_type))) {
        last = *i;
        GetNextChunk(i);
    }
    fseek(i->png, -8, SEEK_CUR);
    *i = last;
    return 0;
}

int PosOverIHDR(png_internal_struct* i) {
    fseek(i->png, 8, SEEK_SET);
    i->length = readLong(i->png);
    fread_s(i->chunk_type, sizeof(i->chunk_type), 1, sizeof(i->chunk_type), i->png);
    i->position = 1;
    return 0;
}

int EndPNG(png_internal_struct* i) {
    if (i->readbuf_initialized) {
        delete i->readbuf;
    }
    return 0;
}


void setChunkType(png_internal_struct* i, const uint8_t* chunkType) {
    memcpy(i->chunk_type, chunkType, sizeof(i->chunk_type));
}

bool checkChunkType(png_internal_struct* i, const uint8_t* chunkType) {
    return 0 == memcmp(i->chunk_type, chunkType, sizeof(i->chunk_type));
}