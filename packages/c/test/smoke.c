/* The C ABI end to end: the version, a refused boot, then (given a model) a
   real one.
     cc test/smoke.c -Iinclude -L<target>/debug -lpie -o smoke && ./smoke [model.zt] */

#include <stdio.h>
#include <string.h>

#include "pie.h"

static int fail(const char *what, pie_status status, char *error) {
    fprintf(stderr, "%s: status %d: %s\n", what, (int)status, error ? error : "(no message)");
    pie_string_free(error);
    return 1;
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
    if (argc < 2) return 0;

    status = pie_server_start(argv[1], "{\"max_total_pages\": 256}", "/tmp/pie-c-smoke", "127.0.0.1:0", &server, &error);
    if (status != PIE_OK) return fail("boot", status, error);
    printf("booted: %s\n", pie_server_summary(server));
    printf("gateway: %s\n", pie_server_listen_addr(server));

    uint32_t session = 0;
    status = pie_server_open_session(server, &session, &error);
    if (status != PIE_OK) return fail("session", status, error);
    printf("session %u\n", session);
    pie_server_close_session(server, session);

    pie_server_shutdown(server);
    status = pie_server_open_session(server, &session, &error);
    if (status != PIE_ERR_SHUT_DOWN) return fail("a session after shutdown", status, error);
    pie_string_free(error);
    printf("shut down\n");
    pie_server_free(server);
    return 0;
}
