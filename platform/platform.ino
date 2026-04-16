#include <Servo.h>

constexpr char DELIMITER = ':';
constexpr int OBJECT_ZONE = 4;
constexpr int MOVE_INTERVAL = 50;
constexpr int MAX_STEP = 4;

struct Pair
{
    int h;
    int v;
};

class ServoController
{
public:
    ServoController(int pin, int object_zone = OBJECT_ZONE, int low_angle = 180, int high_angle = 0, int initial_angle = 90)
        : _pin(pin), _current(initial_angle), _target(initial_angle), _low_angle(low_angle), _high_angle(high_angle), _object_zone(object_zone) {}

    void attach()
    {
        _servo.attach(_pin);
        _servo.write(_current);
    }

    void set_target(int target)
    {
        _target = target;
    }

    int current() const { return _current; }

    bool step()
    {
        int diff = _target - _current;
        if (abs(diff) <= _object_zone)
            return false;

        int s = clamp(diff / 12, -MAX_STEP, MAX_STEP);
        if (s == 0)
            s = (diff > 0) ? 1 : -1;

        _current = clamp(_current + s, _low_angle, _high_angle);
        _servo.write(_current);
        return true;
    }

private:
    Servo _servo;
    int _pin;
    int _current;
    int _target;
    int _low_angle;
    int _high_angle;
    int _object_zone;

    static int clamp(int val, int lo, int hi)
    {
        return (val < lo) ? lo : (val > hi) ? hi
                                            : val;
    }
};

class AngleParser
{
public:
    bool read(Pair &out)
    {
        if (!Serial.available())
            return false;

        int len = Serial.readBytesUntil('\n', _buf, sizeof(_buf) - 1);
        _buf[len] = '\0';

        char *delim = strchr(_buf, DELIMITER);
        if (!delim)
            return false;

        *delim = '\0';
        out.h = atoi(_buf);
        out.v = atoi(delim + 1);
        Serial.write((String(out.h) + " -- " + out.v + "\n").c_str());
        return true;
    }

private:
    char _buf[32];
};

class CameraTracker
{
public:
    CameraTracker()
        : _horizontal(HORIZONTAL_PIN, OBJECT_ZONE , LOW_ANGLES.h,  HIGH_ANGLES.h, 90),
          _vertical(VERTICAL_PIN, OBJECT_ZONE, LOW_ANGLES.v, HIGH_ANGLES.v),
          _last_move(0) {}

    void setup()
    {
        _horizontal.attach();
        _vertical.attach();
    }

    void update()
    {
        Pair coords;
        if (_parser.read(coords))
        {
            _horizontal.set_target(_horizontal.current() + coords.h);
            _vertical.set_target(_vertical.current() + coords.v);
        }

        unsigned long now = millis();
        if (now - _last_move >= MOVE_INTERVAL)
        {
            _horizontal.step() | _vertical.step();
            _last_move = now;
        }
    }

private:
    static constexpr int HORIZONTAL_PIN = 4;
    static constexpr int VERTICAL_PIN = 3;
    static inline constexpr Pair LOW_ANGLES = {0, 39};
    static inline constexpr Pair HIGH_ANGLES = {180, 140};

    ServoController _horizontal;
    ServoController _vertical;
    AngleParser _parser;
    unsigned long _last_move;
};

CameraTracker tracker;

void setup()
{
    Serial.begin(9600);
    tracker.setup();
}

void loop()
{
    tracker.update();
}