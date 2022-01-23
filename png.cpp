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


#include <algorithm>
#include <stdio.h>
#include <stdlib.h>   /* malloc, free */
#include <string.h>
#include <math.h>
#include "png.h"


static uint8_t PaethPredictor(long a, long b, long c);

/* NOTE: In this (new) version of PNG.C the buffer "_readbuf" in */
/* png_internal_struct is allocated via malloc */
/* a call of endPNG() is needed after use of these structure */
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
    uint32_t newCRC = crc;

    if (!crc_table_computed) {
        make_crc_table();
    }
    for (int n = 0; n < len; n++) {
        newCRC = crc_table[(newCRC ^ buf[n]) % 256] ^ (newCRC >> 8);
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


PNGFile::PNGFile(FILE* pngFD, png_info_struct* info) :
    _readbuf(nullptr),
    _buf_filled(0),
    _position(0),
    _chunk_pos(0),
    _zlib_initialized(false)
{
    unsigned char sig[sizeof(png_signature)], infoBuf[100];
    setDefaultInfo();

    _fd = pngFD;
    fseek(_fd, 0L, SEEK_SET);
    fread(sig, 1, sizeof(sig), _fd);
    if (0 != memcmp(sig, png_signature, sizeof(sig))) {
        throw PNGException("Bad signature for PNG file.");
    }
    _length = readLong(_fd);
    fread(_chunk_type, 1, sizeof(_chunk_type), _fd);
    if (0 != memcmp(_chunk_type, image_head_label, sizeof(_chunk_type))) {
        throw PNGException("No PNG header");
    }
    size_t png_info_length = std::max(_length, sizeof(png_info_struct));
    fread_s(infoBuf, sizeof(infoBuf), 1, png_info_length, _fd);
    _info.width = bytes2ulong(&infoBuf[0]);
    _info.height = bytes2ulong(&infoBuf[4]);
    _info.bit_depth = infoBuf[8];
    _info.color_type = infoBuf[9];
    if (2 != _info.color_type) {
        throw PNGException("PNG File must use RGB color.");
    }
    _info.compression = infoBuf[10];
    _info.filter = infoBuf[11];
    _info.interlace = infoBuf[12];
    _crc = readLong(_fd);
    memcpy(info, &_info, sizeof(_info));
}

PNGFile::~PNGFile() {
    if (nullptr != _readbuf) {
        delete _readbuf;
    }
    if (nullptr != _fd) {
        fclose(_fd);
    }
}

bool PNGFile::getNextChunk() {
    if (1 == _position) {
        fseek(_fd, (long)_length + 4, SEEK_CUR);
    } else if (2 == _position) {
        fseek(_fd, 4L, SEEK_CUR);
    }
    _position = 1;
    _length = readLong(_fd);
    return 1 != fread_s(_chunk_type, sizeof(_chunk_type), 1, sizeof(_chunk_type), _fd);
}

int PNGFile::readChunkData(uint8_t* mem) {
    unsigned long checksum;

    if (_position != 1) {
        return -1;
    }
    _mem_ptr = mem;
    fread(_mem_ptr, 1, _length, _fd);
    _crc = readLong(_fd);
    checksum = update_crc(0xffffffffL, (unsigned char*)_chunk_type, sizeof(_chunk_type));
    checksum = update_crc(checksum, _mem_ptr, _length);
    checksum ^= 0xffffffffL;
    if (checksum != _crc) {
        return -2;
    }
    _position = 0;

    return 0;
}

int PNGFile::do_inflate() {
    int err;

    if (!_zlib_initialized) {
        _zlib_initialized = true;
        _d_stream.zalloc = nullptr;
        _d_stream.zfree = nullptr;
        err = inflateInit(&_d_stream);
    }
    err = inflate(&_d_stream, Z_NO_FLUSH);

    if (Z_STREAM_END == err) {
        err = inflateEnd(&_d_stream);
        return 10;
    }
    return err;
}

int PNGFile::readPNGLine(unsigned char* Buf) {
    size_t toread;
    int err;
    int bytesPerPixel;

    /* Test, if chunk is IDAT */
    if (!checkChunkType(image_data_label)) {
        return -1;
    }

    /* Initialize on first call */
    if (0 == _chunk_pos) {
        _checksum = update_crc(0xffffffffL, _chunk_type, sizeof(_chunk_type));
    }
    if (nullptr == _readbuf) {
        _readbuf = new uint8_t[png_bufsize];
    }

    if (0 == _buf_filled) {
        toread = std::max(static_cast<size_t>(_length - _chunk_pos), png_bufsize);
        _buf_filled = fread(_readbuf, 1, toread, _fd);
        if (0 == _buf_filled) {
            return -1;
        }
        _buf_pos = 0;
        _chunk_pos += _buf_filled;
        _checksum = update_crc(_checksum, _readbuf, toread);
    }

    bytesPerPixel = 3 * static_cast<int>(ceil(static_cast<float>(_info.bit_depth) / 8));

    /* Only decompress a single line & filter uint8_t */
    _d_stream.avail_out = static_cast<int>((_info.width * bytesPerPixel + 1));
    _d_stream.next_out = Buf;

    do {
        _d_stream.next_in = _readbuf + _buf_pos;
        _d_stream.avail_in = static_cast<uInt>(_buf_filled - _buf_pos);
        err = do_inflate();
        _buf_pos = _d_stream.next_in - _readbuf;

        /* ready ? */
        if (0 == _d_stream.avail_out) {
            break;
        }
        /* no process made */
        if (_d_stream.avail_out != 0 || _buf_pos == _buf_filled) {
            /* because chunk at end */
            if (_chunk_pos == _length) {
                /* verify crc */
                _checksum ^= 0xffffffffL;
                _crc = readLong(_fd);
                _position = 0;
                if (_checksum != _crc) {
                    return -2;
                }
                /* read next chunk header */
                _length = readLong(_fd);
                fread_s(_chunk_type, sizeof(_chunk_type), sizeof(_chunk_type), 1, _fd);
                _position = 1;
                /* is it an IDAT chunk? */
                if (0 != memcmp(_chunk_type, image_data_label, sizeof(_chunk_type))) {
                    return -3;
                }
                /* initialize _checksum */
                _checksum = update_crc(0xffffffffL, _chunk_type, sizeof(_chunk_type));
                /* prepare reading chunk data */
                _chunk_pos = 0;
                toread = std::max(_length, png_bufsize);
                _buf_filled = fread_s(_readbuf, png_bufsize, 1, toread, _fd);
                if (0 == _buf_filled) {
                    return -1;
                }
                _buf_pos = 0;
                _chunk_pos += _buf_filled;
                _checksum = update_crc(_checksum, _readbuf, toread);
            } else {
                /* because 2K buffer at end */
                toread = std::max(static_cast<size_t>(_length - _chunk_pos), png_bufsize);
                _buf_filled = fread_s(_readbuf, png_bufsize, 1, toread, _fd);
                if (0 == _buf_filled) {
                    return -1;
                }
                _buf_pos = 0;
                _chunk_pos += _buf_filled;
                _checksum = update_crc(_checksum, _readbuf, toread);
            }
        }
    } while (_d_stream.avail_out != 0);

    return 0;
}


int PNGFile::doUnFiltering(uint8_t* Buf, uint8_t* Buf_up) {
    unsigned long bytesPerPixel;
    long up, prior, upperleft;

    bytesPerPixel = 3 * (unsigned long)ceil((float)_info.bit_depth / 8.0);

    switch (Buf[0]) {
    case 0: break;       /* No filter */
    case 1:              /* Sub */
        for (size_t uj = bytesPerPixel + 1; uj <= _info.width * bytesPerPixel; uj++) {
            Buf[uj] = (Buf[uj] + Buf[uj - bytesPerPixel]) % 256;
        }
        Buf[0] = 0;
        break;
    case 2:              /* Up */
        if (nullptr == Buf_up) {
            Buf[0] = 0;
            return 0; /* according to png spec. it is assumed that */
                       /* Buf_up is zero everywhere */
        }
        for (size_t uj = 1; uj <=_info.width * bytesPerPixel; uj++) {
            Buf[uj] = (Buf[uj] + Buf_up[uj]) % 256;
        }
        Buf[0] = 0;
        break;
    case 3:             /* Average */
        for (size_t uj = 1; uj <= _info.width * bytesPerPixel; uj++) {
            if (uj > bytesPerPixel) {
                prior = Buf[uj - bytesPerPixel];
            } else {
                prior = 0;
            }
            if (Buf_up != nullptr) {
                up = Buf_up[uj];
            } else {
                up = 0;
            }
            Buf[uj] = (Buf[uj] + ((prior + up) >> 1)) % 256;
        }
        Buf[0] = 0;
        break;
    case 4:             /* Paeth */
        for (size_t uj = 1; uj <= _info.width * bytesPerPixel; uj++) {
            if (Buf_up != nullptr) {
                up = Buf_up[uj];
            } else {
                up = 0;
            }
            if (uj > bytesPerPixel) {
                prior = Buf[uj - bytesPerPixel];
            } else {
                prior = 0;
            }
            if (uj > bytesPerPixel && Buf_up != nullptr) {
                upperleft = Buf_up[uj - bytesPerPixel];
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

int PNGFile::doFiltering(unsigned char* Buf) {
    if (0 == Buf[0]) {
        return 0; /* No filter */
    }
    if (1 == Buf[0]) {
        /* Sub       */
        int bytesPerPixel = 3 * (int)ceil((double)_info.bit_depth / 8);
        for (int j = (int)_info.width * bytesPerPixel; j >= bytesPerPixel + 1; j--) {
            int t = ((int)Buf[j] - (int)Buf[j - bytesPerPixel]);
            if (t < 0) t += 256;
            Buf[j] = t % 256;
        }
    }
    return 0;
}

int PNGFile::initWritePNG(FILE* png) {
    unsigned long length, crc_v;
    unsigned char Buf[20];

    _fd = png;
    if (fwrite(png_signature, 1, 8, _fd) != 8) {
        return -1;
    }
    _length = length = 13;
    ulong2bytes(length, Buf);
    if (fwrite(Buf, 1, 4, _fd) != 4) {
        return -1;
    }
    if (fwrite(image_head_label, 1, 4, _fd) != 4) {
        return -1;
    }
    ulong2bytes(_info.width, Buf);
    ulong2bytes(_info.height, Buf + 4);
    Buf[8] = _info.bit_depth;
    Buf[9] = _info.color_type;
    Buf[10] = _info.compression;
    Buf[11] = _info.filter;
    Buf[12] = _info.interlace;
    if (fwrite(Buf, 1, 13, png) != 13) {
        return -1;
    }
    crc_v = update_crc(0xffffffffL, (unsigned char*)image_head_label, 4);
    crc_v = update_crc(crc_v, Buf, (long)_length);
    crc_v ^= 0xffffffffL;
    ulong2bytes(crc_v, Buf);
    if (fwrite(Buf, 1, 4, _fd) != 4) {
        return -1;
    }
    memcpy(_chunk_type, image_head_label, sizeof(_chunk_type));
    _chunk_pos = 0;

    return 0;
}

bool PNGFile::writeChunk(unsigned char* buf, size_t size) {
    uint32_t crc_v;
    uint8_t longBuf[sizeof(uint32_t)];

    if ((size_t)-1 == size) {
        _length = size;
    }

    ulong2bytes(static_cast<uint32_t>(_length), longBuf);
    size_t written = fwrite(longBuf, sizeof(longBuf), 1, _fd);
    if (written != 1) {
        return false;
    }
    if (fwrite(_chunk_type, 1, sizeof(_chunk_type), _fd) != sizeof(_chunk_type)) {
        return false;
    }
    if (fwrite(buf, 1, _length, _fd) != _length) {
        return false;
    }
    crc_v = update_crc(0xffffffffL, const_cast<const uint8_t*>(_chunk_type), sizeof(_chunk_type));
    crc_v = update_crc(crc_v, buf, _length);
    crc_v ^= 0xffffffffL;
    ulong2bytes(crc_v, longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, _fd) != 1) {
        return false;
    }
    _position = 0;

    return true;
}

int PNGFile::do_deflate() {
    int err;

    if (!_zlib_initialized) {
        _zlib_initialized = true;
        _d_stream.zalloc = nullptr;
        _d_stream.zfree = nullptr;
        err = deflateInit(&_d_stream, Z_DEFAULT_COMPRESSION);
    }
    err = deflate(&_d_stream, Z_NO_FLUSH);

    return err;
}

int PNGFile::writePNGLine(uint8_t* Buf) {
    int err;
    int bytesPerPixel;

    /* Initialize on first call */
    if (0 == _chunk_pos) {
        _checksum = update_crc(0xffffffffL, image_data_label, sizeof(_chunk_type));
        if (fwrite(&_info, 1, 4, _fd) != 4) {
            return -1;   /* length as dummy */
        }
        if (fwrite("IDAT", 1, 4, _fd) != 4) {
            return -1;
        }
    }
    if (nullptr == _readbuf) {
        _readbuf = new uint8_t[png_bufsize];
    }

    bytesPerPixel = 3 * (int)ceil((double)_info.bit_depth / 8);

    _d_stream.avail_in = (int)((_info.width) * bytesPerPixel + 1); /* Only compress a single line */
    _d_stream.next_in = Buf;

    do {
        _d_stream.next_out = _readbuf;
        _d_stream.avail_out = png_bufsize;
        err = do_deflate();

        _buf_pos = (int)(_d_stream.next_out - _readbuf);
        _checksum = update_crc(_checksum, _readbuf, _buf_pos);
        if (fwrite(_readbuf, 1, _buf_pos, _fd) != _buf_pos) {
            return -1;
        }
        _chunk_pos += _buf_pos;
    } while (_d_stream.avail_out == 0);

    return 0;
}

int PNGFile::endIDAT() {
    int err;
    uint8_t longBuf[sizeof(uint32_t)];

    if (nullptr == _readbuf) {
        return -1;  /* endIDAT called too early */
    }
    do {
        _d_stream.next_out = _readbuf;
        _d_stream.avail_out = png_bufsize;
        err = deflate(&_d_stream, Z_FINISH);
        _buf_pos = (int)(_d_stream.next_out - _readbuf);
        if (fwrite(_readbuf, 1, _buf_pos, _fd) != _buf_pos) {
            return -1;
        }
        _chunk_pos += _buf_pos;
        _checksum = update_crc(_checksum, _readbuf, _buf_pos);
    } while (_d_stream.avail_out == 0);
    err = deflateEnd(&_d_stream);

    _checksum ^= 0xffffffffL;
    ulong2bytes(_checksum, longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, _fd) != 1) {
        return -1;
    }
    fseek(_fd, -(long)(_chunk_pos + 12), SEEK_CUR);
    ulong2bytes(static_cast<uint32_t>(_chunk_pos), longBuf);
    if (fwrite(longBuf, sizeof(longBuf), 1, _fd) != 1) {
        return -1;
    }
    fseek(_fd, static_cast<unsigned long>(_chunk_pos + 8), SEEK_CUR);
    _position = 0;

    return 0;
}

int PNGFile::posOverIEND() {
    PNGFile last;

    while (0 != memcmp(_chunk_type, image_end_label, sizeof(_chunk_type))) {
        if (!getNextChunk()) {
            return 1;
        }
    }
    fseek(_fd, -8, SEEK_CUR);
    return 0;
}

int PNGFile::posOverIHDR() {
    fseek(_fd, 8, SEEK_SET);
    _length = readLong(_fd);
    if (1 == fread_s(_chunk_type, sizeof(_chunk_type), 1, sizeof(_chunk_type), _fd)) {
        _position = 1;
        return 0;
    } else {
        return 1;
    }
}


void PNGFile::setChunkType(const uint8_t* chunkType) {
    memcpy(_chunk_type, chunkType, sizeof(_chunk_type));
}

bool PNGFile::checkChunkType(const uint8_t* chunkType) {
    return 0 == memcmp(_chunk_type, chunkType, sizeof(_chunk_type));
}

void PNGFile::setDefaultInfo() {
    _info.bit_depth = 8;
    _info.color_type = 2;  // RGB color
    _info.compression = 0;
    _info.interlace = 0;
    _info.filter = 0;
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