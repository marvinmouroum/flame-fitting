if platform.system().lower() in ['darwin', 'linux']:
    import sysconfig
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-std=c++14", "-I/opt/homebrew/include/boost"]
    additional_options['extra_compile_args'] = extra_compile_args
    additional_options['library_dirs'] = ['/opt/homebrew/lib']
    additional_options['libraries'] = ['boost_system']

if platform.system().lower() in ['darwin']:
    extra_compile_args+=['-stdlib=libc++']
    extra_link_args=['-stdlib=libc++'] 