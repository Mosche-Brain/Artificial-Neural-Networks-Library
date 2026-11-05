#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <yann.h>

int main(void)
{
    yann_context_t* ctx = NULL;
    yann_model_t* model = NULL;
    yann_loss_t* loss = NULL;
    yann_optimizer_t* opt = NULL;
    yann_tensor_t* X = NULL;
    yann_tensor_t* Y = NULL;
    yann_tensor_t* pred = NULL;
    int exit_code = 1;

    if (yann_create_context_with_device(YANN_DEVICE_CPU, &ctx) != YANN_OK)
    {
        fprintf(stderr, "Failed to create YANN context: %s\n", yann_get_last_error());
        return 1;
    }

    yann_set_verbosity(0);

    if (yann_model_create(ctx, &model) != YANN_OK)
    {
        fprintf(stderr, "Failed to create model: %s\n", yann_get_last_error());
        goto cleanup;
    }

    yann_model_add_input(model, 2);
    yann_model_add_dense(model, 3, "tanh");
    yann_model_add_dense(model, 1, "sigmoid");
    yann_model_build(model, NULL, 0);

    const float x_data[8] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const float y_data[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    if (yann_tensor_create_from_host(ctx, 4, 2, x_data, &X) != YANN_OK ||
        yann_tensor_create_from_host(ctx, 4, 1, y_data, &Y) != YANN_OK)
    {
        fprintf(stderr, "Failed to create data tensors: %s\n", yann_get_last_error());
        goto cleanup;
    }

    if (yann_loss_create_bce(&loss) != YANN_OK ||
        yann_optimizer_create_sgd(0.4f, &opt) != YANN_OK)
    {
        fprintf(stderr, "Failed to create loss/optimizer: %s\n", yann_get_last_error());
        goto cleanup;
    }

    if (yann_model_fit_ex(model, X, Y, loss, opt, 500, 4) != YANN_OK)
    {
        fprintf(stderr, "Training failed: %s\n", yann_get_last_error());
        goto cleanup;
    }

    if (yann_model_predict(model, X, &pred) != YANN_OK)
    {
        fprintf(stderr, "Inference failed: %s\n", yann_get_last_error());
        goto cleanup;
    }

    float out[4] = {0};
    yann_tensor_copy_to_host(pred, out, 4);
    for (int i = 0; i < 4; ++i)
    {
        printf("[%.0f, %.0f] -> %.4f (expected %.0f)\n",
               x_data[i * 2], x_data[i * 2 + 1], out[i], y_data[i]);
        if ((int)roundf(out[i]) != (int)y_data[i])
        {
            fprintf(stderr, "Prediction mismatch at sample %d\n", i);
            goto cleanup;
        }
    }

    exit_code = 0;

cleanup:
    yann_tensor_destroy(pred);
    yann_tensor_destroy(Y);
    yann_tensor_destroy(X);
    yann_optimizer_destroy(opt);
    yann_loss_destroy(loss);
    yann_model_destroy(model);
    yann_destroy_context(ctx);
    return exit_code;
}
