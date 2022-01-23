#include "ComputeWorker.h"
#include "png.h"

void LBuf_ZBuf(
    FractalView& v,
    float* lBuf,
    uint8_t* line,
    int aa_line,
    int offs) {

    for (int k = 0; k < v._antialiasing; k++) {
        for (int i = 0; i < v._xres; i++) {
            int tmp = i * v._antialiasing + k + offs;
            int l = static_cast<int>(floor(lBuf[(i + aa_line * v._xres) * v._antialiasing + k] * 100 + 0.5));;
            line[tmp * 3 + 1] = static_cast<uint8_t>(l >> 16 & 0xFF);
            line[tmp * 3 + 2] = static_cast<uint8_t>(l >> 8 & 0xFF);
            line[tmp * 3 + 3] = static_cast<uint8_t>(l & 0xFF);
        }
    }
}

FractalLineWorker::FractalLineWorker()
    : _lBufR(nullptr), _lBufL(nullptr), _cBuf(nullptr), _bBuf(nullptr), _line(nullptr), _line3(nullptr),
    _rightCalc(nullptr), _leftCalc(nullptr), _rightGPUCalc(nullptr), _leftGPUCalc(nullptr)
{}

FractalLineWorker::FractalLineWorker(
    Quater& quatDriver,
    FractalPreferences& prefs,
    Expression* colorExpr,
    ZFlag zFlag,
    ViewBasis& rbase,
    ViewBasis& srbase,
    ViewBasis& lbase,
    ViewBasis& slbase)
    : _quatDriver(&quatDriver),
    _prefs(prefs), _zFlag(zFlag),
    _srbase(srbase), _slbase(slbase),
    _lBufR(nullptr), _lBufL(nullptr), _cBuf(nullptr), _bBuf(nullptr), _line(nullptr), _line3(nullptr),
    _rightCalc(nullptr), _leftCalc(nullptr), _rightGPUCalc(nullptr), _leftGPUCalc(nullptr),
    _running(false), _failed(false)
{
    _view = prefs.view();
    if (formatExternToIntern(_prefs.fractal(), _view) != 0) {
        throw QuatException("Error in view struct!");
    }
    _xres_st = _view._xres;
    _xres_aa = _view._xres * _view._antialiasing;
    _xres_st_aa = _xres_aa;
    if (_view.isStereo()) {
        _xres_st *= 2;
        _xres_st_aa *= 2;
    }

    try {
        _rightCalc = new LineCalculator(prefs.fractal(), _view, prefs.cuts(), colorExpr, rbase, srbase, zFlag);
        _leftCalc = new LineCalculator(prefs.fractal(), _view, prefs.cuts(), colorExpr, lbase, slbase, zFlag);
        size_t lBufSize = allocBufs();
        _rightGPUCalc = new GPURowCalculator(*_rightCalc, lBufSize, zFlag);
        _leftGPUCalc = new GPURowCalculator(*_leftCalc, lBufSize, zFlag);
    } catch (std::exception&) {
        freeStuff();
        std::rethrow_exception(std::current_exception());
    }
}


FractalLineWorker& FractalLineWorker::operator=(FractalLineWorker&& other) noexcept {
    ComputeWorker::operator=(std::move(other));
    _quatDriver = other._quatDriver;
    _prefs = other._prefs;
    _view = other._view;
    _zFlag = other._zFlag;
    _srbase = other._srbase;
    _slbase = other._slbase;
    _xres_st = other._xres_st;
    _xres_aa = other._xres_aa;
    _xres_st_aa = other._xres_st_aa;

    _lBufR = other._lBufR;
    _lBufL = other._lBufL;
    _cBuf = other._cBuf;
    _bBuf = other._bBuf;
    _line = other._line;
    _line3 = other._line3;
    _rightCalc = other._rightCalc;
    _leftCalc = other._leftCalc;
    _rightGPUCalc = other._rightGPUCalc;
    _leftGPUCalc = other._leftGPUCalc;

    _running = other._running;
    _failed = other._failed;

    other.forgetStuff();
    return *this;
}

size_t FractalLineWorker::allocBufs() {
    FractalView& view = _prefs.view();
    unsigned int st = view.isStereo() ? 2 : 1;
    size_t lBufSize = view._xres * view._antialiasing * (view._antialiasing + 1L);
    size_t paddedLBufSize = KLUDGE_PAD(lBufSize);
    size_t cbBufSize = KLUDGE_PAD(view._xres * st);
    size_t aaLineSize = KLUDGE_PAD(3 * view._xres * view._antialiasing * st + 1);
    _line3 = nullptr;
    switch (_zFlag) {
    case ZFlag::NewImage:   /* create an image without ZBuffer */
        _lBufR = new float[paddedLBufSize];
        _cBuf = new float[cbBufSize];
        _bBuf = new float[cbBufSize];
        _line = new uint8_t[aaLineSize];
        if (2 == st) {
            _lBufL = new float[paddedLBufSize];
        }
        break;
    case ZFlag::NewZBuffer:   /* create a ZBuffer only */
        /* lBufR and lBufL each hold aa lines (mono) */
        _lBufR = new float[paddedLBufSize];
        /* line only holds a single stereo line for transferring of
           lBuf->global ZBuffer */
        _line = new uint8_t[aaLineSize];
        if (2 == st) {
            _lBufL = new float[paddedLBufSize];
        }
        break;
    case ZFlag::ImageFromZBuffer:   /* create an image using a ZBuffer */
        _lBufR = new float[paddedLBufSize];
        _cBuf = new float[cbBufSize];
        _bBuf = new float[cbBufSize];
        _line = new uint8_t[aaLineSize];
        _line3 = new uint8_t[aaLineSize];
        if (2 == st) {
            _lBufL = new float[paddedLBufSize];
        }
        break;
    }
    return paddedLBufSize;
}


FractalLineWorker::~FractalLineWorker() {
    freeStuff();
}

void FractalLineWorker::calcLine(int iy) {
    _failed = false;
    if (iy >= _view._yres) {
        return;
    }
    try {
        _running = true;
        _rightCalc->_xp = _srbase._O + iy * _srbase._y;
        _rightCalc->_xp[3] = _prefs.fractal()._lTerm;
        copyArray(_lBufR, &_lBufR[_view._antialiasing * _xres_aa], _xres_aa);
        if (_view.isStereo()) {
            _leftCalc->_xp = _slbase._O + iy * _slbase._y;
            _leftCalc->_xp[3] = _prefs.fractal()._lTerm;
            copyArray(_lBufL, &_lBufL[_view._antialiasing * _xres_aa], _xres_aa);
        }

        _rightCalc->calcline(*_rightGPUCalc, iy, _lBufR, _bBuf, _cBuf, _zFlag);

        if (shouldCalculateImage(_zFlag) && iy != 0) {
            _prefs.realPalette().pixelValue(0, _view._xres, 255, 255, 255, &_line[1], _cBuf, _bBuf);
        }
        if (_view.isStereo()) {
            _leftCalc->calcline(*_leftGPUCalc, iy, _lBufL, &_bBuf[_view._xres], &_cBuf[_view._xres], _zFlag);
            if (shouldCalculateImage(_zFlag) && iy != 0) {
                _prefs.realPalette().pixelValue(
                    0, _view._xres, 255, 255, 255,
                    &_line[3 * _view._xres + 1],
                    &_cBuf[_view._xres],
                    &_bBuf[_view._xres]);
            }
        }
        _running = false;
    }
    catch (QuatException& ex) {
        _failed_msg = ex.what();
        _failed = true;
        _running = false;
    }
}

void FractalLineWorker::putLine(int iy) {
    if (iy >= _view._yres) {
        return;
    }
    if (ZFlag::NewZBuffer == _zFlag) {
        for (int kk = 1; kk <= _view._antialiasing; kk++) {
            LBuf_ZBuf(_view, _lBufR, _line, kk, 0);
            _quatDriver->putLine(0,
                _view._xres * _view._antialiasing,
                _xres_st_aa,
                iy * _view._antialiasing + kk,
                _line + 1,
                true);
        }
        if (_view.isStereo()) {
            for (int kk = 1; kk <= _view._antialiasing; kk++) {
                LBuf_ZBuf(_view, _lBufL, _line, kk, _xres_aa);
                _quatDriver->putLine(_view._xres,
                    _view._xres * _view._antialiasing + _xres_aa,
                    _xres_st_aa,
                    iy * _view._antialiasing + kk,
                    _line + _view._xres + 1,
                    true);
            }
        }
    } else if (iy > 0) {   /* the image */
        _quatDriver->putLine(0, _view._xres, _xres_st, iy, _line + 1, false);
        if (_view.isStereo()) {
            _quatDriver->putLine(_view._xres, 2 * _view._xres, _xres_st, iy, &_line[3 * _view._xres + 1], false);
        }
    }
}

void FractalLineWorker::writeToPNG(PNGFile* png_internal, int iy) {
    if (iy >= _view._yres) {
        return;
    }
    switch (_zFlag) {
    case ZFlag::NewImage:
        if (iy != 0) {
            _line[0] = 0;       /* Set filter method */
            png_internal->doFiltering(_line);
            png_internal->writePNGLine(_line);
        }
        break;
    case ZFlag::NewZBuffer:
        if (iy != 0) {
            for (int kk = 1; kk < _view._antialiasing + 1; kk++) {
                LBuf_ZBuf(_view, _lBufR, _line, kk, 0);
                if (_view.isStereo()) {
                    LBuf_ZBuf(_view, _lBufL, _line, kk, _xres_aa);
                }
            }
            _line[0] = 0;
            png_internal->doFiltering(_line);
            png_internal->writePNGLine(_line);
        }
        break;
    case ZFlag::ImageFromZBuffer:
        _line[0] = 0;      /* Set filter method */
        png_internal->doFiltering(_line);
        png_internal->writePNGLine(_line);
    }
}

void FractalLineWorker::readZBuffer(int iy) {
    if (iy >= _view._yres) {
        return;
    }
    for (int ii = 0; ii < _view._antialiasing + 1; ii++) {
        if (iy + ii > 0) {  /* this doesn´t work for the 1st line */
            _quatDriver->getline(_line3, iy * _view._antialiasing + ii - 1, _xres_st_aa, ZFlag::NewZBuffer);
            for (int i = 0; i < _xres_aa; i++) {
                _lBufR[i + ii * _xres_aa] = static_cast<float>(threeBytesToLong(&_line[i * 3])) / 100.0f;
                if (_view.isStereo()) {
                    _lBufL[i + ii * _xres_aa] = static_cast<float>(threeBytesToLong(&_line3[(i + _xres_aa) * 3])) / 100.0f;
                }
            }
        } else {
            fillArray(_lBufR, static_cast<float>(_view._zres), _xres_aa);
            if (_view.isStereo()) {
                fillArray(_lBufL, static_cast<float>(_view._zres), _xres_aa);
            }
        }
    }
}


void FractalLineWorker::freeStuff() {
    if (nullptr != _rightCalc) {
        delete _rightCalc;
    }
    if (nullptr != _leftCalc) {
        delete _leftCalc;
    }
    if (nullptr != _lBufR) {
        delete _lBufR;
    }
    if (nullptr != _lBufL) {
        delete _lBufL;
    }
    if (nullptr != _bBuf) {
        delete _bBuf;
    }
    if (nullptr != _cBuf) {
        delete _cBuf;
    }
    if (nullptr != _line) {
        delete _line;
    }
    if (nullptr != _line3) {
        delete _line3;
    }
    if (nullptr != _rightGPUCalc) {
        delete _rightGPUCalc;
    }
    if (nullptr != _leftGPUCalc) {
        delete _leftGPUCalc;
    }
    forgetStuff();
}


void FractalLineWorker::forgetStuff() {
    _lBufR = nullptr;
    _lBufL = nullptr;
    _cBuf = nullptr;
    _bBuf = nullptr;
    _line = nullptr;
    _line3 = nullptr;
    _rightCalc = nullptr;
    _leftCalc = nullptr;
    _rightGPUCalc = nullptr;
    _leftGPUCalc = nullptr;
    _failed = false;
    _running = false;
}