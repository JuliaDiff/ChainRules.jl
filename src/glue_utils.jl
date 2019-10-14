"""
    @version_spec_str(str::String)

Returns a `VersionSpec` object which represents the range of compatible verions,
for a given Pkg3 style semver compat specifier.

Example:
```jldoctest
julia> v"2.0.1" ∈ ChainRules.version_spec"1.2"
false

julia> v"1.5.1" ∈ ChainRules.version_spec"1.2"
true

julia> v"2.0.1" ∈ ChainRules.version_spec"1, 2"
true

julia> v"1.0.1" ∈ ChainRules.version_spec"1, 2"
true

julia> v"1.3.0" ∈ ChainRules.version_spec"~1.0, ~1.1"
false

julia> v"1.1.2" ∈ ChainRules.version_spec"~1.0, ~1.1"
true
"""
macro version_spec_str(str::String)
    return Pkg.Types.semver_spec(str)
end

"""
    pkg_version(_module::Module)

Returns the version of the package that defined a given module.
Does not work on the current module, or on standard libraries
"""
pkg_version(_module::Module)

@static if VERSION ∈ version_spec"~1.0"
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        env = Pkg.Types.Context().env
        pkg_info = Pkg.Types.manifest_info(env, pkg_id.uuid)
        return VersionNumber(pkg_info["version"])
    end
elseif VERSION ∈ version_spec"~1.1"
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        env = Pkg.Types.Context().env
        pkg_info = Pkg.Types.manifest_info(env, pkg_id.uuid)
        return VersionNumber(pkg_info.version)
    end
elseif VERSION ∈ version_spec"~1.2"
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        env = Pkg.Types.Context().env
        pkg_info = Pkg.Types.manifest_info(env, pkg_id.uuid)
        return pkg_info.version
    end
elseif VERSION ∈ version_spec"~1.3"
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        env = Pkg.Types.Context().env
        pkg_info = Pkg.Types.manifest_info(env, pkg_id.uuid)
        return pkg_info.version
    end
else  # tested in 1.4.0-DEV.265
    function pkg_version(_module::Module)
        pkg_id = Base.PkgId(_module)
        ctx = Pkg.Types.Context()
        pkg_info = Pkg.Types.manifest_info(ctx, pkg_id.uuid)
        return pkg_info.version
    end
end
