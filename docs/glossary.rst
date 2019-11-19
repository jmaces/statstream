Glossary
========

.. glossary::
    :sorted:

    streaming data
        Streaming data refers to any kind of data that is not available at once
        and must therefore be processed *on-the-fly*.
        There can be many reasons for this, the two most common being that

        1) the date is generated in real time (for example time-series
        measurements in an experimental setup) and is really simply not
        available at once.

        2) the amount of data is too large to be stored in memory and thus has
        to be processed in a streaming way due to hardware limitations.

    batched data
        When we talk about data we mean numbers organized in arrays, for
        example representing vectors, matrices, tensors, ...
        Generally, streaming data arrives one datum at a time and each datum in
        such a data stream is assumed to be of the same type and shape.
        When we talk about batched data, we mean a stream of data arriving in
        chunks, also called mini-batches.
        For example, if a data stream provides 3-dimensional position vectors
        of shape ``(3,)`` a batched stream would provide groups of shape
        ``(batch_size, 3)`` of multiple such vectors at once.
        The batch dimension is always assumed to be the first. All remaining
        dimensions are referred to as data dimensions.

    singe-pass, one-pass
        A single-pass or one-pass streaming algorithm is one that consumes a
        stream of data exactly once. Such algorithms do not require to restart
        the stream at any time and do not revisited data samples that were
        already seen earlier.
        We do not use the terminology in the strictest way, when dealing with
        batched streaming data. Here one-pass still means that the stream is
        consumed only once. However, as we are processing batches/chunks of
        data at once, we allow multi-pass algorithms per batch. That way each
        data sample can be used multiple times during the algorithm, but the
        batch containing it is only processed once.


    double-pass, two-pass
        A double-pass or two-pass streaming algorithm is one that consumes a
        stream of data exactly twice. Such algorithms assume that the stream of
        data is finite and require to restart the stream and revisit data
        samples that were already seen before in the first pass.
        We do not use the terminology in the strictest way, when dealing with
        batched streaming data. Here two-pass still means that the stream is
        consumed only twice. However, as we are processing batches/chunks of
        data at once, we allow multi-pass algorithms with more than two passes
        per batch. That way each data sample can be used more than twice during
        the algorithm, but the batch containing it is  only processed twice.

    multi-pass, many-pass
        A multi-pass or many-pass streaming algorithm is one that consumes a
        stream of data multiple times. Such algorithms assume that the stream of
        data is finite and require to restart the stream (one or several times)
        and revisit data samples that were already seen before in previous
        passes.
