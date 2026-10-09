/* The C ABI end to end: a refused boot, then (given a model) a real one.
     cc test/smoke.c -Iinclude -L<target>/debug -lpie -o smoke && ./smoke [model.zt] */

#include <stdio.h>
#include "pie.h"

int main(int argc, char **argv) {
    char *error = NULL;
    PieServer *server = pie_server_start("/nonexistent.zt", NULL, "/tmp/pie-c-smoke", NULL, &error);
    if (server != NULL || error == NULL) {
        fprintf(stderr, "a missing artifact booted\n");
        return 1;
    }
    printf("refused: %s\n", error);
    pie_string_free(error);
    if (argc < 2) {
        return 0;
    }

    error = NULL;
    server = pie_server_start(argv[1], NULL, "/tmp/pie-c-smoke", "127.0.0.1:0", &error);
    if (server == NULL) {
        fprintf(stderr, "boot: %s\n", error);
        pie_string_free(error);
        return 1;
    }
    printf("booted: %s\n", pie_server_summary(server));
    printf("gateway: %s\n", pie_server_listen_addr(server));
    uint32_t session = 0;
    if (pie_server_open_session(server, &session, &error) != 0) {
        fprintf(stderr, "session: %s\n", error);
        pie_string_free(error);
        return 1;
    }
    printf("session %u\n", session);
    pie_server_close_session(server, session);
    pie_server_free(server);
    return 0;
}
