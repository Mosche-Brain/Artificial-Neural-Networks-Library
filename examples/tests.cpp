#define ENABLE_DEBUG_OUTPUT
#include <iostream>
#include <iomanip>
#include <cmath>

#define CUM_USE_F16
#define BUILD_USE_MKL
#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/random.hpp>

#include <YANN/Utility/logs.hpp>

static void print_matrix_info(const std::string& name, cum::Matrix& m)
{
    std::cout << name << ": " << m.rows() << " x " << m.cols()
              << " (size=" << m.size() << ")"
              << "  data_ptr=" << static_cast<const void*>(m.data()) << "\n";
}

static void print_matrix_data(const std::string& name, cum::Matrix& m)
{
    print_matrix_info(name, m);
    std::cout << "  data (row-major order): ";
    for (std::size_t i = 0; i < m.size(); ++i) {
        // For f16 we just print the raw bits or try to cast - use float for visibility
        float val = static_cast<float>(m.data()[i]);
        std::cout << std::setw(8) << std::setprecision(4) << val;
        if ((i + 1) % m.cols() == 0) std::cout << " | ";
        else std::cout << " ";
    }
    std::cout << "\n";
}

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    std::cout << "=== MATRIX BEHAVIOUR ANALYSIS (CUM_USE_F16 + row_major MKL) ===\n";
    std::cout << "sizeof(cumeric_t) = " << sizeof(cum::cumeric_t) << " bytes\n\n";

    // =====================================================
    // Test 1: Exact sizes from the failing network
    // Layer 1: weights(3,2) * input(2,1)   -> k=2 (even)
    // Layer 2: weights(1,3) * hidden(3,1)  -> k=3 (odd)  <--- most suspicious
    // =====================================================
    std::cout << "=== Test 1: Network layer sizes (column vector style) ===\n";

    cum::Matrix W1(3, 2, { 0.1_c, 0.2_c,
                           0.3_c, 0.4_c,
                           0.5_c, 0.6_c });
    cum::Matrix x(2, 1, { 0.0_c, 1.0_c });

    cum::Matrix W2(1, 3, { 0.7_c, 0.8_c, 0.9_c });
    cum::Matrix h(3, 1, { 0.11_c, 0.22_c, 0.33_c });

    print_matrix_data("W1 (3x2)", W1);
    print_matrix_data("x  (2x1)", x);

    try {
        cum::Matrix z1 = W1 * x;
        std::cout << "W1 * x  (k=2)  -> ";
        print_matrix_info("z1", z1);
    } catch (const std::exception& e) {
        std::cout << "W1 * x FAILED: " << e.what() << "\n";
    }

    print_matrix_data("W2 (1x3)", W2);
    print_matrix_data("h  (3x1)", h);

    try {
        cum::Matrix z2 = W2 * h;
        std::cout << "W2 * h  (k=3 ODD) -> ";
        print_matrix_info("z2", z2);
    } catch (const std::exception& e) {
        std::cout << "W2 * h (k=3) FAILED: " << e.what() << "\n";
    }

    // =====================================================
    // Test 2: Stress the odd inner dimension (k=3) many times
    // =====================================================
    std::cout << "\n=== Test 2: Stress test k=3 (100 iterations) ===\n";
    bool k3_ok = true;
    for (int i = 0; i < 100; ++i) {
        cum::Matrix a(1, 3);
        cum::Matrix b(3, 1);
        cum::random::uniform(a.data(), a.size(), -1._c, 1._c);
        cum::random::uniform(b.data(), b.size(), -1._c, 1._c);
        try {
            cum::Matrix c = a * b;
            if (i % 25 == 0) {
                std::cout << "  iter " << i << " ok (result ~ " 
                          << static_cast<float>(c.data()[0]) << ")\n";
            }
        } catch (const std::exception& ex) {
            std::cout << "  FAILED at iteration " << i << ": " << ex.what() << "\n";
            k3_ok = false;
            break;
        }
    }
    std::cout << (k3_ok ? "k=3 stress: all 100 passed\n" : "k=3 stress: failed\n");

    // =====================================================
    // Test 3: Backward-style operations that also do matmul
    // deltaWeights = d_pre * inputs^T
    // For output layer: (1x1) * (1x3)   -> k=1
    // For hidden layer: (3x1) * (1x2)   -> k=1
    // =====================================================
    std::cout << "\n=== Test 3: Backward-style matmuls (deltaWeights pattern) ===\n";

    // Output layer case
    cum::Matrix d_out(1, 1, {0.05_c});
    cum::Matrix inpT(1, 3, {0.11_c, 0.22_c, 0.33_c});   // already "transposed" view simulated
    try {
        cum::Matrix dw = d_out * inpT;   // 1x1 * 1x3
        std::cout << "d_out(1x1) * inpT(1x3) succeeded -> " << dw.rows() << "x" << dw.cols() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Output-layer deltaWeights FAILED: " << e.what() << "\n";
    }

    // Hidden layer case (3 neurons, previous layer 2 inputs)
    cum::Matrix d_hidden(3, 1, {0.01_c, 0.02_c, 0.03_c});
    cum::Matrix prevT(1, 2, {0.7_c, 0.8_c});
    try {
        cum::Matrix dw2 = d_hidden * prevT;   // 3x1 * 1x2
        std::cout << "d_hidden(3x1) * prevT(1x2) succeeded -> " << dw2.rows() << "x" << dw2.cols() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Hidden-layer deltaWeights FAILED: " << e.what() << "\n";
    }

    // Also test the gradient propagation: W^T * d_pre
    std::cout << "\n=== Test 4: Gradient backprop matmuls (W^T * d) ===\n";
    cum::Matrix WT(3, 1, {0.7_c, 0.8_c, 0.9_c});   // W2 transposed
    cum::Matrix d(1, 1, {0.05_c});
    try {
        cum::Matrix din = WT * d;   // 3x1 * 1x1
        std::cout << "WT(3x1) * d(1x1) succeeded -> " << din.rows() << "x" << din.cols() << "\n";
    } catch (const std::exception& e) {
        std::cout << "W^T * d FAILED: " << e.what() << "\n";
    }

    // =====================================================
    // Test 5: Row vector input style (what .row(i) produces without transpose)
    // If the network expects column vectors but we feed 1xN, the * will have wrong k.
    // =====================================================
    std::cout << "\n=== Test 5: Row-vector style (1xN) input (common mistake) ===\n";
    cum::Matrix xrow(1, 2, {0.0_c, 1.0_c});
    print_matrix_data("xrow (1x2)", xrow);
    try {
        cum::Matrix bad = W1 * xrow;   // 3x2 * 1x2  -> this is invalid for the intended math
        std::cout << "W1 * xrow (WRONG SHAPE) gave " << bad.rows() << "x" << bad.cols() << "\n";
    } catch (const std::exception& e) {
        std::cout << "W1 * xrow (as expected) error: " << e.what() << "\n";
    }

    // =====================================================
    // Test 6: Even vs Odd k with larger matrices (to see if it's purely odd-k or size related)
    // =====================================================
    std::cout << "\n=== Test 6: Even vs Odd k comparison (f16) ===\n";
    for (int kk : {1, 2, 3, 4, 5, 6, 7, 8}) {
        cum::Matrix Aa(4, kk);
        cum::Matrix Bb(kk, 2);
        cum::random::uniform(Aa.data(), Aa.size(), -0.5_c, 0.5_c);
        cum::random::uniform(Bb.data(), Bb.size(), -0.5_c, 0.5_c);
        try {
            cum::Matrix Cc = Aa * Bb;
            std::cout << "  4x" << kk << " * " << kk << "x2  -> OK (result 4x2)\n";
        } catch (const std::exception& e) {
            std::cout << "  4x" << kk << " * " << kk << "x2  -> FAILED: " << e.what() << "\n";
        }
    }

    // Simulate a simple sigmoid for testing (not using YANN's functions here)
    (void)0; // placeholder so we don't have unused lambda warning if we comment things out

    auto approx_sigmoid_deriv_inplace = [](cum::Matrix& out, cum::Matrix& preact) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            float v = static_cast<float>(preact.data()[i]);
            float s = 1.0f / (1.0f + std::exp(-v));
            float d = s * (1.0f - s);
            out.data()[i] = static_cast<cum::cumeric_t>(d);
        }
    };

    // =====================================================
    // Test 7: REPLICATE THE EXACT TRAINING LOOP PATTERN (with stored inputs like Dense)
    // This mirrors Sequential::fit + Dense forward/backward/update exactly.
    // We store 'inputs' per layer (as Dense does), use them in backward with .transpose(),
    // reset deltas to zero-matrices, do the arithmetic updates, then continue to next sample.
    // The goal is to reproduce the 'lda illegal' on hgemm with f16 + row_major.
    // =====================================================
    {
    std::cout << "\n=== Test 7: Exact Dense-like cycle (4 samples) ===\n";

    // Persistent layer state (like Dense members)
    cum::Matrix W1(3, 2), b1(3, 1), dW1(3, 2, 0_c), db1(3, 1, 0_c);
    cum::Matrix W2(1, 3), b2(1, 1), dW2(1, 3, 0_c), db2(1, 1, 0_c);
    cum::Matrix inputs1, inputs2;   // what Dense stores in forward

    cum::random::uniform(W1.data(), W1.size(), -0.5_c, 0.5_c);
    cum::random::uniform(b1.data(),  b1.size(),  -0.1_c, 0.1_c);
    cum::random::uniform(W2.data(), W2.size(), -0.5_c, 0.5_c);
    cum::random::uniform(b2.data(),  b2.size(),  -0.1_c, 0.1_c);

    cum::Matrix X(4, 2, {0,0, 0,1, 1,0, 1,1});
    cum::Matrix Y(4, 1, {0, 1, 1, 0});

    cum::cumeric_t rate = 0.01_c;

    auto reset_delta = [](cum::Matrix& d, std::size_t r, std::size_t c) {
        d = cum::Matrix(r, c, 0_c);
    };

    bool loop_ok = true;
    for (int sample = 0; sample < 4 && loop_ok; ++sample) {
        std::cout << "  Sample " << sample << "\n";

        // column vector input (like after .row(i).transpose())
        cum::Matrix x(2, 1);
        x.data()[0] = X.data()[sample * 2 + 0];
        x.data()[1] = X.data()[sample * 2 + 1];
        cum::Matrix target(1, 1);
        target.data()[0] = Y.data()[sample];

        try {
            // ----- LAYER 1 FORWARD (like Dense::forward) -----
            inputs1 = x;                                   // store inputs (2x1)
            std::cout << "    [fwd] inputs1=" << inputs1.rows() << "x" << inputs1.cols()
                      << "  W1=" << W1.rows() << "x" << W1.cols() << " (k=" << W1.cols() << ")\n";
            cum::Matrix pre1 = (W1 * inputs1) + b1;        // k=2
            cum::Matrix a1(3, 1);
            for (std::size_t j=0; j<pre1.size(); ++j) {
                float v = static_cast<float>(pre1.data()[j]);
                a1.data()[j] = static_cast<cum::cumeric_t>(1.0f/(1.0f+std::exp(-v)));
            }
            std::cout << "    L1 fwd OK (pre1 3x1, k=2)\n";

            // ----- LAYER 2 FORWARD (k=3, this is where original crash often hits on sample 1) -----
            inputs2 = a1;                                  // store hidden as inputs (3x1)
            std::cout << "    [fwd] inputs2=" << inputs2.rows() << "x" << inputs2.cols()
                      << "  W2=" << W2.rows() << "x" << W2.cols() << " (k=" << W2.cols() << ")\n";
            cum::Matrix pre2 = (W2 * inputs2) + b2;        // 1x3 * 3x1, k=3
            cum::Matrix a2(1, 1);
            for (std::size_t j=0; j<pre2.size(); ++j) {
                float v = static_cast<float>(pre2.data()[j]);
                a2.data()[j] = static_cast<cum::cumeric_t>(1.0f/(1.0f+std::exp(-v)));
            }
            std::cout << "    L2 fwd OK (pre2 1x1, k=3)\n";

            // ----- loss grad (MSE) -----
            cum::Matrix d_out(1, 1);
            d_out.data()[0] = static_cast<cum::cumeric_t>(
                2.0f * (static_cast<float>(a2.data()[0]) - static_cast<float>(target.data()[0])) );

            // ----- LAYER 2 BACKWARD (exactly like Dense) -----
            cum::Matrix dpre2(1, 1);
            approx_sigmoid_deriv_inplace(dpre2, pre2);
            dpre2.data()[0] = static_cast<cum::cumeric_t>(
                static_cast<float>(dpre2.data()[0]) * static_cast<float>(d_out.data()[0]) );

            // deltaW = dpre * inputs^T   (1x1 * 1x3)
            auto t_in2 = inputs2.transpose();
            std::cout << "    [bwd L2] dpre2=" << dpre2.rows() << "x" << dpre2.cols()
                      << "  inputs2.T=" << t_in2.rows() << "x" << t_in2.cols() << " (will use k=" << dpre2.cols() << ")\n";
            cum::Matrix dW2_new = dpre2 * t_in2;
            cum::Matrix db2_new = dpre2;

            // gradient to previous = W^T * dpre
            auto t_w2 = W2.transpose();
            std::cout << "    [bwd L2] W2.T=" << t_w2.rows() << "x" << t_w2.cols()
                      << "  dpre2=" << dpre2.rows() << "x" << dpre2.cols() << "\n";
            cum::Matrix dhidden = t_w2 * dpre2;   // 3x1 * 1x1

            std::cout << "    L2 bwd OK (used inputs2.transpose() and W2.transpose())\n";

            // ----- LAYER 1 BACKWARD -----
            cum::Matrix dpre1(3, 1);
            approx_sigmoid_deriv_inplace(dpre1, pre1);
            for (std::size_t j=0; j<3; ++j) {
                float g = static_cast<float>(dpre1.data()[j]) * static_cast<float>(dhidden.data()[j]);
                dpre1.data()[j] = static_cast<cum::cumeric_t>(g);
            }

            auto t_in1 = inputs1.transpose();
            std::cout << "    [bwd L1] dpre1=" << dpre1.rows() << "x" << dpre1.cols()
                      << "  inputs1.T=" << t_in1.rows() << "x" << t_in1.cols() << " (k for this mul=" << dpre1.cols() << ")\n";
            cum::Matrix dW1_new = dpre1 * t_in1;  // 3x1 * 1x2
            cum::Matrix db1_new = dpre1.rowwiseSum();

            std::cout << "    L1 bwd OK (used inputs1.transpose() and rowwiseSum)\n";

            // ----- UPDATE (like Dense::update_weights) -----
            dW2 = dW2_new;
            db2 = db2_new;
            dW1 = dW1_new;
            db1 = db1_new;

            std::cout << "    [update] about to do W2 = W2 - (dW2 * rate);  dW2="
                      << dW2.rows() << "x" << dW2.cols() << "  W2=" << W2.rows() << "x" << W2.cols() << "\n";
            W2 = W2 - (dW2 * rate);
            b2 = b2 - (db2 * rate);
            W1 = W1 - (dW1 * rate);
            b1 = b1 - (db1 * rate);

            std::cout << "    [update] about to reset deltas\n";
            reset_delta(dW2, 1, 3);
            reset_delta(db2, 1, 1);
            reset_delta(dW1, 3, 2);
            reset_delta(db1, 3, 1);

            std::cout << "    update+reset OK\n";

            // Critical: right after reset, try the exact shape that will be used on the next sample's L2 forward
            cum::Matrix probe(3, 1);
            cum::random::uniform(probe.data(), probe.size(), -0.2_c, 0.2_c);
            std::cout << "    [probe] W2=" << W2.rows() << "x" << W2.cols()
                      << "  probe=" << probe.rows() << "x" << probe.cols() << " (k=" << W2.cols() << ")\n";
            try {
                cum::Matrix z = W2 * probe;   // 1x3 * 3x1  <--- this is the shape that blew up in original run
                std::cout << "    post-update k=3 probe mul OK\n";
            } catch (const std::exception& ex) {
                std::cout << "    !!! post-update k=3 probe FAILED: " << ex.what() << "\n";
            }

        } catch (const std::exception& ex) {
            std::cout << "  !!! FAILED during sample " << sample << ": " << ex.what() << "\n";
            loop_ok = false;
        }
    }

    std::cout << (loop_ok ? "Test 7: completed 4 samples (check if any post-update probe failed)\n"
                        : "Test 7: hit error during the cycle\n");

    // =====================================================
    // Test 8: Isolate transpose + mul after the exact update sequence
    // =====================================================
    std::cout << "\n=== Test 8: Transpose-involved muls after update ===\n";
    // recreate a fresh W2-like and go through one update cycle, then hammer transpose muls
    cum::Matrix W(1, 3);
    cum::Matrix dW(1, 3, 0_c);
    cum::random::uniform(W.data(), W.size(), -0.5_c, 0.5_c);
    cum::Matrix inp(3, 1);
    cum::random::uniform(inp.data(), inp.size(), -0.3_c, 0.3_c);
    cum::Matrix dpre(1, 1, {0.1_c});

    // simulate one backward step
    cum::Matrix delta = dpre * inp.transpose();   // 1x1 * 1x3
    W = W - (delta * rate);
    dW = cum::Matrix(1, 3, 0_c);  // reset like Dense

    bool t_ok = true;
    for (int i=0; i<30 && t_ok; ++i) {
        cum::Matrix v(3, 1);
        cum::random::uniform(v.data(), v.size(), -1._c, 1._c);
        try {
            // This is the pattern from L2 gradient backprop: W.transpose() * something
            cum::Matrix g = W.transpose() * dpre;
            // and also the deltaW pattern
            cum::Matrix dw = dpre * v.transpose();  // but v would be previous layer act
            if (i % 10 == 0) std::cout << "  transpose-mul iter " << i << " OK\n";
        } catch (const std::exception& ex) {
            std::cout << "  transpose-mul FAILED at iter " << i << ": " << ex.what() << "\n";
            t_ok = false;
        }
    }
    std::cout << (t_ok ? "Test 8: transpose muls after update OK\n" : "Test 8: transpose mul blew up\n");

    // =====================================================
    // Test 9: Try to reproduce the exact original timing (forward after update on sample 1)
    // Do two full (fwd, bwd, update) cycles and attempt L2 fwd on the second pass.
    // =====================================================
    std::cout << "\n=== Test 9: Two full cycles, focusing on second L2 forward ===\n";
    cum::Matrix WW1(3,2), bb1(3,1), WW2(1,3), bb2(1,1);
    cum::random::uniform(WW1.data(), WW1.size(), -0.4_c, 0.4_c);
    cum::random::uniform(bb1.data(), bb1.size(), -0.05_c, 0.05_c);
    cum::random::uniform(WW2.data(), WW2.size(), -0.4_c, 0.4_c);
    cum::random::uniform(bb2.data(), bb2.size(), -0.05_c, 0.05_c);

    cum::Matrix i1, i2;
    bool second_fwd_ok = true;
    for (int s=0; s<2 && second_fwd_ok; ++s) {
        cum::Matrix xx(2,1, { (float)(s%2), (float)((s+1)%2) });
        try {
            i1 = xx;
            cum::Matrix p1 = (WW1 * i1) + bb1;
            cum::Matrix a1(3,1);
            for (std::size_t j=0;j<p1.size();++j) a1.data()[j] = static_cast<cum::cumeric_t>(1.0f/(1.0f+std::exp(-static_cast<float>(p1.data()[j]))));

            i2 = a1;
            cum::Matrix p2 = (WW2 * i2) + bb2;   // k=3
            std::cout << "  cycle " << s << " L2 fwd OK\n";

            // minimal backward + update
            cum::Matrix dout(1,1, {0.1_c});
            cum::Matrix dp2(1,1);
            approx_sigmoid_deriv_inplace(dp2, p2);
            dp2.data()[0] = static_cast<cum::cumeric_t>(static_cast<float>(dp2.data()[0]) * 0.1f);

            cum::Matrix dww2 = dp2 * i2.transpose();
            WW2 = WW2 - (dww2 * rate);
            // reset delta (not really used here)
            dW2 = cum::Matrix(1,3,0_c);

            // now on the next iteration the WW2 will be used for fwd again
        } catch (const std::exception& ex) {
            std::cout << "  cycle " << s << " FAILED: " << ex.what() << "\n";
            second_fwd_ok = false;
        }
    }
    // After the first cycle's update, explicitly try the L2 forward shape again (simulating sample 1 L2)
    if (second_fwd_ok) {
        try {
            cum::Matrix probe(3,1);
            cum::random::uniform(probe.data(), probe.size(), -0.1_c, 0.1_c);
            cum::Matrix z = WW2 * probe;
            std::cout << "  explicit L2 fwd after first update OK\n";
        } catch (const std::exception& ex) {
            std::cout << "  explicit L2 fwd after update FAILED: " << ex.what() << "\n";
            second_fwd_ok = false;
        }
    }
    std::cout << (second_fwd_ok ? "Test 9: two cycles + post-update fwd OK\n" : "Test 9: reproduced lda error in second fwd or after update\n");
    }

    // =====================================================
    // Test 10: MINIMAL REPRO for odd-column left matrix (the smoking gun for lda)
    // In row-major + hgemm, when left matrix A has odd number of columns (lda = A.cols() = odd),
    // the row_major::gemm call with lda=odd tends to be rejected on Arc GPUs.
    // =====================================================
    std::cout << "\n=== Test 10: Minimal odd-lda repro (left matrix with odd #cols) ===\n";

    // Case that just failed in Test 7: 3x1 * 1x2   (lda for left = 1, which is odd)
    {
        cum::Matrix left(3, 1);   // 3 rows, 1 col -> row stride 1 (odd)
        cum::Matrix right(1, 2);
        for (std::size_t i=0; i<left.size(); ++i) left.data()[i] = static_cast<cum::cumeric_t>(0.1f * (i+1));
        for (std::size_t i=0; i<right.size(); ++i) right.data()[i] = static_cast<cum::cumeric_t>(0.2f * (i+1));
        std::cout << "  left 3x1 (cols=1 odd), right 1x2\n";
        try {
            cum::Matrix r = left * right;
            std::cout << "    3x1 * 1x2 -> OK (unexpected if this is the failing path)\n";
        } catch (const std::exception& ex) {
            std::cout << "    3x1 * 1x2 -> CRASHED as expected: " << ex.what() << "\n";
        }
    }

    // The original network's L2 forward pattern: 1x3 * 3x1   (lda for left = 3, odd)
    {
        cum::Matrix left(1, 3);   // 1 row, 3 cols -> row stride 3 (odd)
        cum::Matrix right(3, 1);
        for (std::size_t i=0; i<left.size(); ++i) left.data()[i] = static_cast<cum::cumeric_t>(0.3f + 0.1f*i);
        for (std::size_t i=0; i<right.size(); ++i) right.data()[i] = static_cast<cum::cumeric_t>(0.4f + 0.1f*i);
        std::cout << "  left 1x3 (cols=3 odd), right 3x1\n";
        try {
            cum::Matrix r = left * right;
            std::cout << "    1x3 * 3x1 -> OK\n";
        } catch (const std::exception& ex) {
            std::cout << "    1x3 * 3x1 -> CRASHED: " << ex.what() << "\n";
        }
    }

    // Control: even column counts
    {
        cum::Matrix left(3, 2);   // cols=2 even
        cum::Matrix right(2, 2);
        for (std::size_t i=0; i<left.size(); ++i) left.data()[i] = static_cast<cum::cumeric_t>(0.01f * i);
        for (std::size_t i=0; i<right.size(); ++i) right.data()[i] = static_cast<cum::cumeric_t>(0.02f * i);
        std::cout << "  control: left 3x2 (cols=2 even)\n";
        try {
            cum::Matrix r = left * right;
            std::cout << "    3x2 * 2x2 -> OK (even cols)\n";
        } catch (const std::exception& ex) {
            std::cout << "    3x2 * 2x2 -> FAILED: " << ex.what() << "\n";
        }
    }

    // Another dangerous pattern from backward: 3x1 * 1x1  (lda=1 odd)  -- this is W^T * dpre for output layer
    {
        cum::Matrix left(3, 1);
        cum::Matrix right(1, 1);
        left.data()[0] = 0.1_c; left.data()[1] = 0.2_c; left.data()[2] = 0.3_c;
        right.data()[0] = 0.05_c;
        std::cout << "  left 3x1 (cols=1), right 1x1  -- W^T * delta pattern\n";
        try {
            cum::Matrix r = left * right;
            std::cout << "    3x1 * 1x1 -> OK\n";
        } catch (const std::exception& ex) {
            std::cout << "    3x1 * 1x1 -> CRASHED: " << ex.what() << "\n";
        }
    }

    // =====================================================
    // Test 11: Check if padding the rightmost dimension to even avoids the crash
    // (This would be a workaround inside cum::Matrix or the matmul wrapper)
    // =====================================================
    std::cout << "\n=== Test 11: Workaround idea - pad odd column counts to even stride ===\n";
    // We can't easily change cum::Matrix stride from here, but we can emulate what a padded
    // row-major buffer would look like by using a larger buffer and manual indexing.
    // For now just document that lda must be even for f16 row_major on this device.
    std::cout << "  (Analysis only) On this Arc B580 + oneMKL row_major + f16, lda (== logical cols for non-transposed left matrix)\n"
              << "  appears to be rejected when odd. Workaround: ensure all Matrix 'cols' are even (pad with zeros),\n"
              << "  or switch the LinearAlgebra::matMul to a path that doesn't hit the strict hgemm lda check (e.g. use\n"
              << "  a temporary transposed copy, or call into a different MKL path, or pad at allocation time).\n";

    std::cout << "\n=== End of matrix behaviour tests ===\n";
    cum::decum();
    return 0;
}
