/**
 * Grammar Language VS Code Extension
 */

import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';

import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
  // The server is implemented in node
  const serverModule = context.asAbsolutePath(
    path.join('..', 'tools', 'lsp-server.ts')
  );

  // The debug options for the server
  // --inspect=6009: runs the server in Node's Inspector mode so VS Code can attach to the server for debugging
  const debugOptions = { execArgv: ['--nolazy', '--inspect=6009'] };

  // If the extension is launched in debug mode then the debug server options are used
  // Otherwise the run options are used
  const serverOptions: ServerOptions = {
    run: {
      module: serverModule,
      transport: TransportKind.ipc,
      runtime: 'tsx'
    },
    debug: {
      module: serverModule,
      transport: TransportKind.ipc,
      options: debugOptions,
      runtime: 'tsx'
    }
  };

  // Options to control the language client
  const clientOptions: LanguageClientOptions = {
    // Register the server for Grammar Language documents
    documentSelector: [{ scheme: 'file', language: 'grammar-language' }],
    synchronize: {
      // Notify the server about file changes to '.gl' files contained in the workspace
      fileEvents: workspace.createFileSystemWatcher('**/*.gl')
    }
  };

  // Create the language client and start the client
  client = new LanguageClient(
    'grammarLanguageServer',
    'Grammar Language Server',
    serverOptions,
    clientOptions
  );

  // Start the client. This will also launch the server
  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
