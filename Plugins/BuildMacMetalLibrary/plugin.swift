import Foundation
import PackagePlugin

@main struct BuildMacMetalLibrary: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        let root = context.package.directoryURL
        let source = root.appending(path: "Source/Cmlx/mlx-generated/metal")
        let builder = root.appending(path: "tools/build-macos-metallib.sh")
        let library = context.pluginWorkDirectoryURL.appending(path: "default.metallib")
        guard let files = FileManager.default.enumerator(
            at: source, includingPropertiesForKeys: [.isRegularFileKey])
        else { throw CocoaError(.fileReadNoSuchFile) }
        let inputs = try files.compactMap { item -> URL? in
            guard let file = item as? URL,
                try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
            else { return nil }
            return file
        }.sorted { $0.path < $1.path }

        return [.buildCommand(
            displayName: "Compile macOS Metal kernels without fast math",
            executable: URL(fileURLWithPath: "/bin/zsh"),
            arguments: [builder.path, source.path, library.path],
            inputFiles: [builder] + inputs,
            outputFiles: [library])]
    }
}
