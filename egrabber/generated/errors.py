# Copyright Euresys 2021

"""Generated exceptions that translate GenTL error codes."""

from . import constants as c

class GenTLException(Exception):
    def __init__(self, message, error):
        super(GenTLException, self).__init__(message)
        self.gc_err = error

class TimeoutException(GenTLException):
    def __init__(self):
        super(TimeoutException, self).__init__(_strerrors[c.GC_ERR_TIMEOUT], c.GC_ERR_TIMEOUT)

class errorCheck:

    def __init__(self, f, n):
        self.f = f
        self.n = n

    def __call__(self, *a, **kw):
        res = self.f(*a, **kw)
        if res == c.GC_ERR_TIMEOUT:
            raise TimeoutException()
        elif res != 0 and res in _strerrors:
            raise GenTLException(_strerrors[res], res)
        elif res != 0:
            raise GenTLException("GenTL error %d" % res, res)
        return res

# Errcodes translation
_strerrors = {
    c.GC_ERR_SUCCESS : "GC_ERR_SUCCESS",
    c.GC_ERR_ERROR : "GC_ERR_ERROR",
    c.GC_ERR_NOT_INITIALIZED : "GC_ERR_NOT_INITIALIZED",
    c.GC_ERR_NOT_IMPLEMENTED : "GC_ERR_NOT_IMPLEMENTED",
    c.GC_ERR_RESOURCE_IN_USE : "GC_ERR_RESOURCE_IN_USE",
    c.GC_ERR_ACCESS_DENIED : "GC_ERR_ACCESS_DENIED",
    c.GC_ERR_INVALID_HANDLE : "GC_ERR_INVALID_HANDLE",
    c.GC_ERR_INVALID_ID : "GC_ERR_INVALID_ID",
    c.GC_ERR_NO_DATA : "GC_ERR_NO_DATA",
    c.GC_ERR_INVALID_PARAMETER : "GC_ERR_INVALID_PARAMETER",
    c.GC_ERR_IO : "GC_ERR_IO",
    c.GC_ERR_TIMEOUT : "GC_ERR_TIMEOUT",
    c.GC_ERR_ABORT : "GC_ERR_ABORT",
    c.GC_ERR_INVALID_BUFFER : "GC_ERR_INVALID_BUFFER",
    c.GC_ERR_NOT_AVAILABLE : "GC_ERR_NOT_AVAILABLE",
    c.GC_ERR_INVALID_ADDRESS : "GC_ERR_INVALID_ADDRESS",
    c.GC_ERR_BUFFER_TOO_SMALL : "GC_ERR_BUFFER_TOO_SMALL",
    c.GC_ERR_INVALID_INDEX : "GC_ERR_INVALID_INDEX",
    c.GC_ERR_PARSING_CHUNK_DATA : "GC_ERR_PARSING_CHUNK_DATA",
    c.GC_ERR_INVALID_VALUE : "GC_ERR_INVALID_VALUE",
    c.GC_ERR_RESOURCE_EXHAUSTED : "GC_ERR_RESOURCE_EXHAUSTED",
    c.GC_ERR_OUT_OF_MEMORY : "GC_ERR_OUT_OF_MEMORY",
    c.GC_ERR_BUSY : "GC_ERR_BUSY",
    c.GC_ERR_CUSTOM_ID : "GC_ERR_CUSTOM_ID",
    c.GC_ERR_CUSTOM_MULTIPLE_HANDLES : "GC_ERR_CUSTOM_MULTIPLE_HANDLES",
    c.GC_ERR_CUSTOM_DANGLING_HANDLES : "GC_ERR_CUSTOM_DANGLING_HANDLES",
    c.GC_ERR_CUSTOM_LOST_HANDLE : "GC_ERR_CUSTOM_LOST_HANDLE",
    c.GC_ERR_CUSTOM_LOCK_ERROR : "GC_ERR_CUSTOM_LOCK_ERROR",
    c.GC_ERR_CUSTOM_SILENT_ERROR : "GC_ERR_CUSTOM_SILENT_ERROR",
    c.GC_ERR_CUSTOM_LIBPATH_ERROR : "GC_ERR_CUSTOM_LIBPATH_ERROR",
    c.GC_ERR_CUSTOM_DRIVER_NOT_AVAILABLE : "GC_ERR_CUSTOM_DRIVER_NOT_AVAILABLE",
    c.GC_ERR_CUSTOM_DRIVER_IO_ERROR : "GC_ERR_CUSTOM_DRIVER_IO_ERROR",
    c.GC_ERR_CUSTOM_REVOKE_ERROR_FOLLOWING_ANNOUNCE_ERROR : "GC_ERR_CUSTOM_REVOKE_ERROR_FOLLOWING_ANNOUNCE_ERROR",
    c.GC_ERR_CUSTOM_STD_EXCEPTION : "GC_ERR_CUSTOM_STD_EXCEPTION",
    c.GC_ERR_CUSTOM_ALIGNMENT_ERROR : "GC_ERR_CUSTOM_ALIGNMENT_ERROR",
    c.GC_ERR_CUSTOM_WAIT_FAILED : "GC_ERR_CUSTOM_WAIT_FAILED",
    c.GC_ERR_CUSTOM_WAIT_INTERRUPTED : "GC_ERR_CUSTOM_WAIT_INTERRUPTED",
    c.GC_ERR_CUSTOM_CANNOT_CREATE_NOTIFIER : "GC_ERR_CUSTOM_CANNOT_CREATE_NOTIFIER",
    c.GC_ERR_CUSTOM_NOTIFIER_ERROR : "GC_ERR_CUSTOM_NOTIFIER_ERROR",
    c.GC_ERR_CUSTOM_LOADING_ERROR : "GC_ERR_CUSTOM_LOADING_ERROR",
    c.GC_ERR_CUSTOM_SYMBOL_NOT_FOUND : "GC_ERR_CUSTOM_SYMBOL_NOT_FOUND",
    c.GC_ERR_CUSTOM_STRING_TOO_LONG : "GC_ERR_CUSTOM_STRING_TOO_LONG",
    c.GC_ERR_CUSTOM_DATATYPE_MISMATCH : "GC_ERR_CUSTOM_DATATYPE_MISMATCH",
    c.GC_ERR_CUSTOM_TOO_MANY_GENAPI_CONTEXTS : "GC_ERR_CUSTOM_TOO_MANY_GENAPI_CONTEXTS",
    c.GC_ERR_CUSTOM_INCORRECT_OEM_SAFETY_KEY : "GC_ERR_CUSTOM_INCORRECT_OEM_SAFETY_KEY",
    c.GC_ERR_CUSTOM_OPAQUE_NETWORK : "GC_ERR_CUSTOM_OPAQUE_NETWORK",
    c.GC_ERR_CUSTOM_GENAPI_FEATURE_NOT_FOUND : "GC_ERR_CUSTOM_GENAPI_FEATURE_NOT_FOUND",
    c.GC_ERR_CUSTOM_GENAPI_ERROR : "GC_ERR_CUSTOM_GENAPI_ERROR",
    c.GC_ERR_CUSTOM_IMAGE_ERROR : "GC_ERR_CUSTOM_IMAGE_ERROR",
    c.GC_ERR_CUSTOM_LICENSE_MANAGER_ERROR : "GC_ERR_CUSTOM_LICENSE_MANAGER_ERROR",
    c.GC_ERR_CUSTOM_NO_LICENSE : "GC_ERR_CUSTOM_NO_LICENSE",
    c.GC_ERR_CUSTOM_IOCTL_BASE : "GC_ERR_CUSTOM_IOCTL_BASE",
    c.GC_ERR_CUSTOM_IOCTL_PCI_WRITE_CONFIG_FAILED : "GC_ERR_CUSTOM_IOCTL_PCI_WRITE_CONFIG_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_PCI_READ_CONFIG_FAILED : "GC_ERR_CUSTOM_IOCTL_PCI_READ_CONFIG_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DS_CREATE_NO_DMA_ENGINE : "GC_ERR_CUSTOM_IOCTL_DS_CREATE_NO_DMA_ENGINE",
    c.GC_ERR_CUSTOM_IOCTL_DS_CREATE_NO_IRQ_HANDLER : "GC_ERR_CUSTOM_IOCTL_DS_CREATE_NO_IRQ_HANDLER",
    c.GC_ERR_CUSTOM_IOCTL_DS_REGISTER_EVENT_FAILED : "GC_ERR_CUSTOM_IOCTL_DS_REGISTER_EVENT_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DS_GET_INFO_FAILED : "GC_ERR_CUSTOM_IOCTL_DS_GET_INFO_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DS_OPEN_FAILED : "GC_ERR_CUSTOM_IOCTL_DS_OPEN_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DS_START_FAILED : "GC_ERR_CUSTOM_IOCTL_DS_START_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DS_STOP_FAILED : "GC_ERR_CUSTOM_IOCTL_DS_STOP_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DEV_GET_INFO_FAILED : "GC_ERR_CUSTOM_IOCTL_DEV_GET_INFO_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_DEV_GET_URL_INFO_FAILED : "GC_ERR_CUSTOM_IOCTL_DEV_GET_URL_INFO_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_CXP_CONNECTION_WRITE_FAILED : "GC_ERR_CUSTOM_IOCTL_CXP_CONNECTION_WRITE_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_CXP_CONNECTION_READ_FAILED : "GC_ERR_CUSTOM_IOCTL_CXP_CONNECTION_READ_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_CXP_MASTER_CONNECTION_NOT_FOUND : "GC_ERR_CUSTOM_IOCTL_CXP_MASTER_CONNECTION_NOT_FOUND",
    c.GC_ERR_CUSTOM_IOCTL_CXP_HOST_LIBRARY_CMD_FAILED : "GC_ERR_CUSTOM_IOCTL_CXP_HOST_LIBRARY_CMD_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_BUFFER_ANNOUNCE_FAILED : "GC_ERR_CUSTOM_IOCTL_BUFFER_ANNOUNCE_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_STOP_TIMEOUT : "GC_ERR_CUSTOM_IOCTL_STOP_TIMEOUT",
    c.GC_ERR_CUSTOM_IOCTL_I2C_FAILED : "GC_ERR_CUSTOM_IOCTL_I2C_FAILED",
    c.GC_ERR_CUSTOM_IOCTL_BANK_SELECT_INCONSISTENCY : "GC_ERR_CUSTOM_IOCTL_BANK_SELECT_INCONSISTENCY",
    c.GC_ERR_CUSTOM_IOCTL_ONBOARD_MEMORY_READ_ERROR : "GC_ERR_CUSTOM_IOCTL_ONBOARD_MEMORY_READ_ERROR",
    c.GC_ERR_CUSTOM_IOCTL_ONBOARD_MEMORY_WRITE_ERROR : "GC_ERR_CUSTOM_IOCTL_ONBOARD_MEMORY_WRITE_ERROR",
    c.GC_ERR_CUSTOM_IOCTL_FFC_WRITE_ERROR : "GC_ERR_CUSTOM_IOCTL_FFC_WRITE_ERROR",
    c.GC_ERR_CUSTOM_IOCTL_SERIAL_REGISTER_EVENT_FAILED : "GC_ERR_CUSTOM_IOCTL_SERIAL_REGISTER_EVENT_FAILED",
    c.GC_ERR_CUSTOM_GEV_BASE : "GC_ERR_CUSTOM_GEV_BASE",
}
