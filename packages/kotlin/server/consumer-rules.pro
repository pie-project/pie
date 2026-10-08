# NativeCore's methods are bound by name from libpie_server.so, and the core
# throws PieException.Server by name.
-keep class org.pieproject.server.NativeCore { native <methods>; }
-keep class org.pieproject.client.PieException$Server { <init>(java.lang.String); }
