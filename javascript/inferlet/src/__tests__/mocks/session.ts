// Alias target for 'pie:inferlet/session' under vitest. See ../stubs.ts.
import { sessionStub } from '../stubs.js';
export const { send, receive, receiveBlocking, sendFile, receiveFile, receiveFileBlocking } = sessionStub;
