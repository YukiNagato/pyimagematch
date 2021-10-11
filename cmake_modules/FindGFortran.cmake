SET(QUADMATH_NAMES quadmath)
SET(GFORTRAN_NAMES gfortran libgfortran)

FIND_LIBRARY(QUADMATH_LIBRARY
  NAMES ${QUADMATH_NAMES}
  PATHS /usr/lib64/atlas /usr/lib/atlas 
        /usr/lib64 /usr/lib /usr/local/lib64 
        /usr/local/lib /usr/x86_64-linux-gnu/*
        /usr/lib/gcc/x86_64-linux-gnu/*
  )

FIND_LIBRARY(GFORTRAN_LIBRARY
  NAMES ${GFORTRAN_NAMES}
  PATHS /usr/lib64/atlas /usr/lib/atlas 
        /usr/lib64 /usr/lib /usr/local/lib64 
        /usr/local/lib /usr/x86_64-linux-gnu/*
        /usr/lib/gcc/x86_64-linux-gnu/*
  )

IF (QUADMATH_LIBRARY)
    SET(QUADMATH_LIBRARIES ${QUADMATH_LIBRARY})
    SET(QUADMATH_FOUND "YES")
    MESSAGE(STATUS "Found QUADMATH: ${QUADMATH_LIBRARIES}")
ELSE (QUADMATH_LIBRARY)
    SET(QUADMATH_FOUND "NO")
    IF (QUADMATH_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find QuadMath")
    ENDIF (QUADMATH_FIND_REQUIRED)
ENDIF (QUADMATH_LIBRARY)

IF (GFORTRAN_LIBRARY)
    SET(GFORTRAN_LIBRARIES ${GFORTRAN_LIBRARY})
    SET(GFORTRAN_FOUND "YES")
    MESSAGE(STATUS "Found GFORTRAN: ${GFORTRAN_LIBRARIES}")
ELSE (GFORTRAN_LIBRARY)
    SET(GFORTRAN_FOUND "NO")
    IF (GFORTRAN_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Gfortran")
    ENDIF (GFORTRAN_FIND_REQUIRED)
ENDIF (GFORTRAN_LIBRARY)
