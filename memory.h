#pragma once

#include <cerrno>
#include <cassert>
#include <stdio.h>
#include <stdexcept>

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
    ~LexicallyScopedPtr() {
        if (nullptr != _ptr) {
            delete _ptr;
            _ptr = nullptr;
        }
    }
    T* operator=(T* ptr) {
        if (nullptr != _ptr) {
            delete _ptr;
        }
        _ptr = ptr;
        return _ptr;
    }
    operator T* () { return _ptr; }
    T* operator->() {
        return &*_ptr;
    }
    const T* ptr() const {
        return _ptr;
    }
    T* ptr() {
        return _ptr;
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
        _mem = nullptr;
    }
    const T* ptr() const {
        return _mem;
    }
    T* ptr() {
        return _mem;
    }
    T& operator[](size_t i) {
        if (i >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::operator[]");
        }
        return _mem[i];
    }
    T& operator[](size_t i) const {
        if (i >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::operator[] const");
        }
        return _mem[i];
    }
    void setElems(size_t offset, const T* ptr, size_t nToCopy) {
        if (offset + nToCopy >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::setElems");
        }
        memcpy(&_mem[offset], ptr, nToCopy * sizeof(T));
    }
    void getElems(T* ptr, size_t offset, size_t bToCopy) const {
        if (offset + nToCopy >= _nElems) {
            throw std::out_of_range("Index out of range for LexicallyScopedRangeCheckedStorage::getElems");
        }
        memcpy(ptr, &_mem[offset], nToCopy * sizeof(T));
    }
};

template<typename T>
inline void fillArray(T* array, size_t nElems, const T& value) {
    for (size_t i = 0; i < nElems; i++) {
        array[i] = value;
    }
}
