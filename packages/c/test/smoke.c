/* The C ABI end to end: the version, a refused boot, then (given models) a
   real one of each, one after another in this process, each asked one chat
   completion through its gateway and shut down before the next boots: a
   shutdown frees the process for the next model.
     cc test/smoke.c -Iinclude -L<target>/debug -lpie -o smoke && ./smoke [model.zt...] */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "pie.h"

static int fail(const char *what, pie_status status, char *error) {
    fprintf(stderr, "%s: status %d: %s\n", what, (int)status, error ? error : "(no message)");
    pie_string_free(error);
    return 1;
}

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* What the Metal device holds for this process, in bytes; -1 off Apple. Read
   through the library's own links, so the test links nothing more. */
static long long device_bytes(void) {
#ifdef __APPLE__
    void *(*device)(void) = (void *(*)(void))dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
    void *(*sel)(const char *) = (void *(*)(const char *))dlsym(RTLD_DEFAULT, "sel_registerName");
    void *send = dlsym(RTLD_DEFAULT, "objc_msgSend");
    if (!device || !sel || !send) return -1;
    void *mtl = device();
    if (!mtl) return -1;
    return (long long)((unsigned long (*)(void *, void *))send)(mtl, sel("currentAllocatedSize"));
#else
    return -1;
#endif
}

/* One chat completion through the gateway at `addr` (host:port): its reply,
   printed, and whether it answered 200. */
static int complete(const char *addr) {
#ifdef _WIN32
    (void)addr;
    return 0;
#else
    char host[64];
    const char *colon = strrchr(addr, ':');
    if (!colon || (size_t)(colon - addr) >= sizeof host) return 1;
    memcpy(host, addr, (size_t)(colon - addr));
    host[colon - addr] = '\0';
    struct sockaddr_in to = {.sin_family = AF_INET, .sin_port = htons((uint16_t)atoi(colon + 1))};
    if (inet_pton(AF_INET, host, &to.sin_addr) != 1) return 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0 || connect(fd, (struct sockaddr *)&to, sizeof to) != 0) return 1;
    const char *body = "{\"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Answer in one word.\"}], "
                       "\"max_tokens\": 16, \"temperature\": 0}";
    char request[1024];
    int n = snprintf(request, sizeof request,
                     "POST /v1/chat/completions HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\n"
                     "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
                     addr, strlen(body), body);
    if (write(fd, request, (size_t)n) != n) return 1;
    static char response[1 << 16];
    size_t got = 0;
    ssize_t r;
    while (got < sizeof response - 1 && (r = read(fd, response + got, sizeof response - 1 - got)) > 0) got += (size_t)r;
    close(fd);
    response[got] = '\0';
    char *reply = strstr(response, "\r\n\r\n");
    printf("completion: %s\n", reply ? reply + 4 : response);
    return strncmp(response, "HTTP/1.1 200", 12) != 0;
#endif
}

int main(int argc, char **argv) {
    if (strcmp(pie_version(), PIE_VERSION_STRING) != 0) {
        fprintf(stderr, "pie.h is %s, the library %s\n", PIE_VERSION_STRING, pie_version());
        return 1;
    }

    pie_server *server = NULL;
    char *error = NULL;
    pie_status status = pie_server_start("/nonexistent.zt", NULL, "/tmp/pie-c-smoke", NULL, &server, &error);
    if (status == PIE_OK || server != NULL || error == NULL) {
        fprintf(stderr, "a missing artifact booted\n");
        return 1;
    }
    printf("refused (%d): %s\n", (int)status, error);
    pie_string_free(error);
    error = NULL;

    status = pie_server_start(NULL, NULL, "/tmp/pie-c-smoke", NULL, &server, &error);
    if (status != PIE_ERR_INVALID_ARGUMENT) return fail("a NULL artifact", status, error);
    pie_string_free(error);
    error = NULL;

    for (int i = 1; i < argc; i++) {
        printf("boot %d: %s\ndevice bytes before boot: %lld\n", i, argv[i], device_bytes());
        double start = now();
        status = pie_server_start(argv[i], "{\"budget_device\": \"1GiB\"}", "/tmp/pie-c-smoke", "127.0.0.1:0", &server, &error);
        if (status != PIE_OK) return fail("boot", status, error);
        printf("booted in %.2f s: %s\n", now() - start, pie_server_summary(server));
        printf("gateway: %s\ndevice bytes serving: %lld\n", pie_server_listen_addr(server), device_bytes());

        uint32_t session = 0;
        status = pie_server_open_session(server, &session, &error);
        if (status != PIE_OK) return fail("session", status, error);
        printf("session %u\n", session);
        pie_server_close_session(server, session);
        /* The model's own faults are not the ABI's: printed, not fatal. */
        if (complete(pie_server_listen_addr(server)) != 0) printf("the completion failed\n");

        pie_server_shutdown(server);
        status = pie_server_open_session(server, &session, &error);
        if (status != PIE_ERR_SHUT_DOWN) return fail("a session after shutdown", status, error);
        pie_string_free(error);
        error = NULL;
        pie_server_free(server);
        server = NULL;
        printf("shut down; device bytes after: %lld\n", device_bytes());
    }
    return 0;
}
