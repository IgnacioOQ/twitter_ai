#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>
#include <graphviz/types.h>

static int fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: sfdp_coords INPUT.dot OUTPUT.f64 SEED\n");
        return 2;
    }

    FILE *input = fopen(argv[1], "rb");
    if (input == NULL) {
        fprintf(stderr, "cannot open input: %s\n", strerror(errno));
        return 1;
    }
    Agraph_t *graph = agread(input, NULL);
    fclose(input);
    if (graph == NULL) {
        return fail("Graphviz could not parse the input graph");
    }

    agsafeset(graph, "dim", "3", "3");
    agsafeset(graph, "dimen", "3", "3");
    agsafeset(graph, "start", argv[3], argv[3]);
    agsafeset(graph, "overlap", "true", "true");
    agsafeset(graph, "splines", "false", "false");

    GVC_t *context = gvContext();
    if (context == NULL) {
        agclose(graph);
        return fail("Could not create a Graphviz context");
    }
    if (gvLayout(context, graph, "sfdp") != 0) {
        gvFreeContext(context);
        agclose(graph);
        return fail("sfdp layout failed");
    }

    const size_t node_count = (size_t)agnnodes(graph);
    if (GD_ndim(graph) != 3) {
        gvFreeLayout(context, graph);
        gvFreeContext(context);
        agclose(graph);
        return fail("sfdp did not produce a three-dimensional layout");
    }
    double *coordinates = calloc(node_count * 3, sizeof(double));
    unsigned char *seen = calloc(node_count, sizeof(unsigned char));
    if (coordinates == NULL || seen == NULL) {
        free(coordinates);
        free(seen);
        gvFreeLayout(context, graph);
        gvFreeContext(context);
        agclose(graph);
        return fail("Could not allocate the coordinate output buffer");
    }

    size_t found = 0;
    for (Agnode_t *node = agfstnode(graph); node != NULL; node = agnxtnode(graph, node)) {
        char *end = NULL;
        errno = 0;
        const unsigned long long parsed = strtoull(agnameof(node), &end, 10);
        if (errno != 0 || end == agnameof(node) || *end != '\0' || parsed >= node_count) {
            free(coordinates);
            free(seen);
            gvFreeLayout(context, graph);
            gvFreeContext(context);
            agclose(graph);
            return fail("Node names must be contiguous zero-based integers");
        }
        const size_t index = (size_t)parsed;
        if (seen[index]) {
            free(coordinates);
            free(seen);
            gvFreeLayout(context, graph);
            gvFreeContext(context);
            agclose(graph);
            return fail("Duplicate numeric node name");
        }
        const double *position = ND_pos(node);
        if (position == NULL) {
            free(coordinates);
            free(seen);
            gvFreeLayout(context, graph);
            gvFreeContext(context);
            agclose(graph);
            return fail("A node has no internal position");
        }
        coordinates[index * 3] = position[0];
        coordinates[index * 3 + 1] = position[1];
        coordinates[index * 3 + 2] = position[2];
        seen[index] = 1;
        found++;
    }
    if (found != node_count) {
        free(coordinates);
        free(seen);
        gvFreeLayout(context, graph);
        gvFreeContext(context);
        agclose(graph);
        return fail("Did not collect every node position");
    }

    FILE *output = fopen(argv[2], "wb");
    if (output == NULL) {
        free(coordinates);
        free(seen);
        gvFreeLayout(context, graph);
        gvFreeContext(context);
        agclose(graph);
        fprintf(stderr, "cannot open output: %s\n", strerror(errno));
        return 1;
    }
    const size_t values = node_count * 3;
    const size_t written = fwrite(coordinates, sizeof(double), values, output);
    const int close_result = fclose(output);

    free(coordinates);
    free(seen);
    gvFreeLayout(context, graph);
    gvFreeContext(context);
    agclose(graph);
    if (written != values || close_result != 0) {
        return fail("Could not write the complete coordinate payload");
    }
    fprintf(stderr, "wrote %zu nodes in %u dimensions\n", node_count, 3u);
    return 0;
}
