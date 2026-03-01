# Speculative Decoding

Speculative Decoding is an optimization algorithm that helps increase throughput of LLM generation as the auto-regressive generation is outputs one token at a time.


## Definition from Paper

Speculative Decoding - an algorithm to sample autoregressive models faster without any change to the outputs, by computing several tokens in parallel. 

At the heart of our approach lie two observations -

1. Hard Language Modeling tasks often include easier subtasks that can be approximated well by more efficient models.

2. Using Speculative Execution and a novel sampling method, we can make decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently and without changing the distribution.

This method can accelerate existing off-the-shelf models without retraining or architecture changes.

## Mathematics 

### Overview

Let M<sub>p</sub> be the target model, we are trying to accelerate and the probability distribution of <math>p(x|x<sub>< t</sub>)</math> <math>{or p(x) }</math> for a prefix of <math> x<sub> < t </sub>.</math>

Let M<sub>q</sub> be the more efficient approximation model and the probability distribution of <math>q(x|x<sub>< t</sub>)</math> <math>{or q(x) }</math> for a prefix of <math> x<sub> < t </sub>.</math>

The core idea is to 
1. Use the more efficient model M<sub>q</sub> to generate γ ∈ Z<sup> + </sup> completions. (We will talk about choosing this parameter later).
2. Use the target model M<sub>p</sub> to evaluate all the guesses and their respective probabilities from M<sub>q</sub> in parallel, accepting those that lead to identical distributions.
3. Sampling an additional token from an adjusted distribution to fix the first one that was rejected, or to add an additional one if they are all accepted. (New or Corrected token from M<sub>p</sub> model).

This way, each parallel run of the target model M<sub>p</sub> will produce at least one new token (so the number of serial runs of the target model can never, even in the worst case, be larger than the simple autoregressive method), but it can potentially generate many new tokens, up to γ + 1, depending on how well M<sub>q</sub> approximates M<sub>p</sub>.

<b> In Layman Terms </b> : M<sub>q</sub> model is used to generations for γ steps and then M<sub>p</sub> model is used to judge the generations, it is then used to either correct the first rejection or add a new token if all generations are accepted. This in worst case still faster than equal to γ serial runs of M<sub>p</sub> model run.

### Standard Sampling

There are many methods and parameters of sampling, like argmax, nucleus, top-k and setting a temperature and popular implementations, treat them differently at logits level, they can easily be cast into standard sampling from an adjusted probability distribution.

For example, argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalising. We can therefore only deal with standard sampling from a probability distribution and cast all the other types of sampling into that framework.
Going forward we’ll assume that p(x) and q(x) are the distributions from M<sub>p</sub> and M<sub>q</sub> respectively, adjusted for the sampling method.

<b> In Layman Terms </b> : All tricks used in inference of autoregressive models for speed or consistency can be turned into a probability distribution adjusted with the sampling method, making it easier for comparison purposes.

### Speculative Sampling

To sample x ~ <math>p(x)</math>, we instead sample x ~ <math>q(x)</math> keeping <math> q(x) ≤ p(x) </math>, in case <math> q(x) > p(x) </math> we reject the sample with probability (1 - p(x)) and sample x again from q(x) adjusted distribution with <math> p′(x) = norm(max(0, p(x) − q(x))) </math> instead.

Note: It is also shown in the paper (Appendix A.1) that for any distributions p(x) and q(x), and x sampled in this way, indeed x ∼ p(x).

Given the distribution q(x) obtained from running M<sub>q</sub> on a conditioning prefix, we can sample a token x1 ∼ q(x). We then calculate the distribution p(x) by running M<sub>p</sub> on prefix while in parallel speculatively calculating the distribution of the next token x<sub>2</sub> by running M<sub>p</sub> on <i>prefix+[x1]</i>.

Once both computations complete, we proceed with: If x<sub>1</sub> is rejected, we discard the computation of x<sub>2</sub> and re-sample x<sub>1</sub> from an adjusted distribution, and if x<sub>1</sub> is accepted, we keep both tokens. Algorithm 1 generalizes this idea to sample between 1 and γ + 1 tokens at once.


<b>Algorithm 1 SpeculativeDecodingStep </b>

    Inputs: Mp, Mq, prefix.

    ◃ Sample γ guesses x1,...,γ from Mq autoregressively.
    for i = 1 to γ do

        qi(x) ← Mq(prefix + [x1, . . . , xi−1])
        xi ∼ qi(x) end for
    ◃ Run Mp in parallel. 

    p1(x),...,pγ+1(x) ← Mp(prefix),...,Mp(prefix+[x1,...,xγ])

    ◃ Determine the number of accepted guesses n.
    r1 ~ U(0,1),...,rγ ∼ U(0,1)
    n ←- min({i−1|1 ≤ i≤ γ,ri > pi(x)/qi(x)}∪{γ})

    ◃ Adjust the distribution from Mp if needed. 
    p′(x) ← pn+1(x)
    if n < γ then
        p′(x) ← norm(max(0, pn+1(x) − qn+1(x))) 
    end if
    
    ◃ Return one token from Mp, and n tokens from Mq.
    t ∼ p′(x)
    return prefix + [x1, . . . , xn, t]



## Analysis

### Number of Generated Tokens

### Calculating Alpha

### Walltime Improvements

### Number of Arithmetic Operations

### Choosing Gamma

### Approximation Models