#pragma once

#include <cerrno>
#include <stdio.h>
#include <stdexcept>
#include <type_traits>
#include "CReplacements.h"

class LexicallyScopedFile {
    FILE* _fd;
    errno_t _errno;
public:
    LexicallyScopedFile() : _fd(nullptr), _errno(0) {}
    LexicallyScopedFile(FILE* fd) : _fd(fd), _errno(0) {}
    LexicallyScopedFile(const char* const name, const char* const mode) {
        _errno = fopen_s(&_fd, name, mode);
    }
    ~LexicallyScopedFile() {
        if (nullptr != _fd) {
            assert(0 == fflush(_fd));
            fclose(_fd);
        }
    }
    FILE* operator=(FILE* fd) {
        assert(nullptr == _fd);
        _fd = fd;
        return _fd;
    }
    operator FILE* () {
        return _fd;
    }
    bool isOpen() const {
        return nullptr != _fd;
    }
    errno_t error() const {
        return _errno;
    }
};


template<typename T>
class LexicallyScopedPtr {
    T* _ptr;
public:
    LexicallyScopedPtr() : _ptr(nullptr) {}
    LexicallyScopedPtr(T* ptr) : _ptr(ptr) {}
    LexicallyScopedPtr(size_t size) {
        _ptr = new T[size];
    }
    ~LexicallyScopedPtr() {
        freeContained();
    }
    T* operator=(T* ptr) {
        freeContained();
        _ptr = ptr;
        return _ptr;
    }
    operator T* () {
        return _ptr;
    }
    operator const T* () const {
        return _ptr;
    }
    T* operator->() {
        return &*_ptr;
    }
    const T* ptr() const {
        return _ptr;
    }
    T* ptr() {
        return _ptr;
    }
private:
    void freeContained() {
        if (nullptr != _ptr) {
            if (std::is_array<T>()) {
                delete[] _ptr;
            } else {
                delete _ptr;
            }
            _ptr = nullptr;
        }
    }
};


template<typename T>
class LexicallyScopedRangeCheckedStorage {
    T* _mem;
    size_t _nElems;
public:

    typedef T Type;

    explicit LexicallyScopedRangeCheckedStorage(size_t nElems) : _nElems(nElems) {
        _mem = new T[nElems]();
    }
    explicit LexicallyScopedRangeCheckedStorage(size_t nElems, const T* elems) : _nElems(nElems) {
        setElems(0, elems, nElems);
    }
    ~LexicallyScopedRangeCheckedStorage() {
        delete []_mem;
    }
    const T* ptr() const {
        return _mem;
    }
    T* ptr() {
        return _mem;
    }
    T& operator[](size_t i) {
#ifndef NDEBUG
        if (i >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::operator[]");
        }
#endif
        return _mem[i];
    }
    T& operator[](size_t i) const {
#ifndef NDEBUG
        if (i >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::operator[] const");
        }
#endif
        return _mem[i];
    }
    void setElems(size_t offset, const T* ptr, size_t nToCopy) {
#ifndef NDEBUG
        if (offset + nToCopy >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::setElems");
        }
#endif
        memcpy(&_mem[offset], ptr, nToCopy * sizeof(T));
    }
    void getElems(T* ptr, size_t offset, size_t nToCopy) const {
#ifndef NDEBUG
        if (offset + nToCopy >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::getElems");
        }
#endif
        memcpy(ptr, &_mem[offset], nToCopy * sizeof(T));
    }
};