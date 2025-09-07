import enum

class UserRole(str, enum.Enum):
    admin = "admin"
    officer = "officer"
    citizen = "citizen"
    device = "device"  # pseudo-role, not for login via /auth/login

class DeviceType(str, enum.Enum):
    ip_camera = "ip_camera"
    officer_mobile = "officer_mobile"
