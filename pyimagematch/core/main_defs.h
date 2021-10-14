#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined CVAPI_EXPORTS
#  define CV_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CV_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define CV_EXPORTS
#endif