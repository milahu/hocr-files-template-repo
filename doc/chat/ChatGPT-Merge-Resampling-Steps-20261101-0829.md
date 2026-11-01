# Merge Resampling Steps

**User:** Anonymous  
**Created:** 2026/11/1 8:27:49  
**Updated:** 2026/11/1 8:28:10  
**Exported:** 2026/11/1 8:29:16  
**Link:** [<https://chatgpt.com/c/6ae6e9f4-a9c4-83eb-8a5b-95c67387946a>](https://chatgpt.com/c/6ae6e9f4-a9c4-83eb-8a5b-95c67387946a)

## Prompt:

11/1/2026, 8:27:47 AM

in image processing, how bad is the "generation loss" from multiple resampling steps?  
in my book-scanning pipeline (`https://github.com/milahu/hocr-files-template-repo`) i currently have two steps where i resample the scanned images: `065-remove-page-borders.py` and `070-deskew.py`. both steps "deskew" (rotate and transform) the scanned images. the difference is, `065-remove-page-borders.py` uses the page edges (to rotate the scanned image so the page edges are as horizontal and vertical as possible) and `070-deskew.py` uses the page contents (to rotate the scanned image so the lines of text are as horizontal as possible).  
question: should i remove my `070-deskew.py` step for better output quality? (or if horizontal text lines are important, then merge both steps into one, to minimize the damage/noise/loss from resampling)

## Response:

11/1/2026, 8:27:51 AM

> **Searched 14 websites**
>
> Worked for 10s

# Short answer

**I would merge `065-remove-page-borders.py` and `070-deskew.py` into a single geometric resampling step, rather than simply removing `070-deskew.py`.**

Your intuition is correct: repeatedly resampling a scanned image can degrade it. But the damage from a second rotation is usually modest compared with the benefit of correcting the text's orientation, especially for OCR.

The key distinction is between:

- **Estimating the geometry:** detecting page edges and text baselines. These operations can happen independently, on the original image or on temporary analysis copies.
- **Applying the geometry:** rotating, translating, scaling, or warping the pixels. Ideally, combine these transformations and resample the original pixels just once.

If your current pipeline first rotates to align the page edges, then rotates again to align the text, it is doing two interpolations where one would normally suffice.

## 1. How bad is the generation loss?

There isn't a single percentage that describes it. The loss depends on the rotation angles, image resolution, interpolation kernel, source image type, and whether any cropping or downsampling occurs.

For a typical book scan, the practical effects are:

- **Slight blurring:** repeated interpolation can soften small character details and thin strokes.
- **Pixel-value changes:** new pixels are calculated from neighboring pixels rather than retaining the original samples.
- **Possible aliasing and artifacts:** fine details can be distorted, depending on the resampling filter.
- **Irrecoverable cropping:** pixels discarded at an intermediate step cannot be recovered by a later transform.

These are established reasons to avoid repeated resampling; image-processing references recommend composing successive transformations and resampling only once. \<Cite refs={\["turn275112search2","turn275112search1","turn275112search11"\]}/\>

### Is the second rotation especially damaging?

Consider an illustrative scan with two small corrections:

- First rotation: $`+0.7^\circ`$
- Second rotation: $`-0.4^\circ`$

The net rotation is only $`+0.3^\circ`$.

If both transformations are applied sequentially, the image is interpolated twice. If they are combined, the original pixels are interpolated once at the net angle.

The difference is particularly relevant for small text, thin serifs, punctuation, and faint characters. But the exact loss depends on the image and implementation. Two small rotations don't necessarily cause severe visible degradation, and one rotation at a larger angle is not automatically better in every respect.

One important detail: **the damage is not proportional simply to the number of steps.** Two rotations of an already-resampled image are generally worse than one equivalent rotation from the original, but the magnitude varies.

## 2. In your pipeline, both deskew estimates have a purpose

The two steps solve different problems:

| Step | What it estimates | Why it matters |
|----|----|----|
| `065-remove-page-borders.py` | Page geometry and edges | Correct page alignment, edge cleanup, cropping and border removal |
| `070-deskew.py` | Text-line orientation | Horizontal text baselines and better OCR layout |

The important question isn't which step is more correct. It's whether their geometric corrections need to be applied separately.

A physical page edge and the text printed on it are not guaranteed to have precisely the same orientation. A page can be slightly distorted, the printed text can be imperfectly aligned, or the page-edge detector can be noisy. Consequently, removing the text-based correction just to preserve image quality could leave a real skew in the text.

For OCR, correcting text-line skew is a standard preprocessing operation. OCRmyPDF, for example, includes deskewing specifically to correct pages scanned at an angle. \<Cite refs={\["turn275112search24","turn275112search9"\]}/\>

## 3. What I would change in your implementation

I'd aim for this pipeline:

<box border radius="lg" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3} align="center">
    **Original scanned image**

    <text color="secondary" size="sm">Keep the original pixel data intact</text>

  </box>
  <box align="center">
    <icon name="arrow-down" size="lg" color="secondary"/>
  </box>
  <box border radius="md" padding={3} gap={2}>
    <title size="md">Analyze the original image</title>
    <row align="start" gap={2}>
      <icon name="scan" size="lg" color="secondary"/>
      <box flex="1" gap={1}>
        **Estimate page geometry**

        <text color="secondary" size="sm">Find the page edges, borders and desired crop.</text>
      </box>
    </row>
    <divider color="subtle"/>
    <row align="start" gap={2}>
      <icon name="align-horizontal-justify-center" size="lg" color="secondary"/>
      <box flex="1" gap={1}>
        **Estimate text skew**

        <text color="secondary" size="sm">Find the dominant text-line orientation, using the original image or a temporary analysis copy.</text>
      </box>
    </row>

  </box>
  <box align="center">
    <icon name="arrow-down" size="lg" color="secondary"/>
  </box>
  <box background="surface" border radius="md" padding={3} align="center" gap={1}>
    <icon name="combine" size="xl"/>
    **Combine the geometric transformations**

    <text color="secondary" size="sm" textAlign="center">Calculate the final mapping, crop and output dimensions.</text>

  </box>
  <box align="center">
    <icon name="arrow-down" size="lg" color="secondary"/>
  </box>
  <box background="rgba(22,163,74,0.10)" border={{size:1,color:"rgba(22,163,74,0.35)"}} radius="md" padding={3} align="center" gap={1}>
    <icon name="check-circle" size="xl" color="success"/>
    **Resample the original once**

    <text color="secondary" size="sm" textAlign="center">Write the final image with the desired geometry and margins.</text>

  </box>
</box>

This preserves the benefits of both steps without unnecessarily applying their transformations sequentially.

### A few implementation details

**1. Estimate both corrections before modifying the pixels.**

The text-skew detector does not need to consume the already-rotated output of `065`. It can estimate the text angle from the original image or from a temporary image prepared specifically for detection.

**2. Compose the transformations mathematically.**

If both corrections are rotations, their angles can be combined, with appropriate care for their coordinate systems and rotation centers.

If `065` performs a more general affine or perspective transformation, combine the corresponding transformation matrices or mappings instead. For nonlinear warps, compose the mappings and resample the original once.

This is the same general principle used in image-registration workflows. \<Cite refs={\["turn275112search2","turn275112search11"\]}/\>

**3. Combine the crop with the final transformation where practical.**

If you can determine the final output bounds in advance, you can avoid intermediate rotations that create unwanted corners or discard image regions. Make sure the final crop doesn't cut off page content.

**4. Preserve intermediate images only for debugging, not as the source for the next geometric step.**

You can still save intermediate visualizations to inspect the page-edge detection and text-angle estimation. Just don't feed those resampled visualizations into the next transformation when the original pixels are available.

## 4. What about interpolation filters?

If you combine the steps, the interpolation filter used for the final transformation still matters.

For grayscale or color scans, bicubic interpolation is a reasonable starting point. Higher-quality filters may help in some cases, but the best choice depends on whether you prioritize fine detail, sharpness, or avoiding ringing around dark text on a light background. Pillow documents the available filters and their different interpolation behaviors. \<Cite refs={\["turn275112search1","turn275112search4"\]}/\>

A few caveats:

- Avoid nearest-neighbor interpolation for ordinary grayscale scans when smooth geometric transformations are needed; it tends to produce jagged edges.
- Avoid repeatedly converting between grayscale, binary and color representations during the geometric operations.
- If you have a binary, strictly black-and-white image, interpolation deserves separate consideration because intermediate gray values may be introduced. In many pipelines, it is preferable to deskew the grayscale image before binarizing it.
- Avoid JPEG encoding and decoding between processing steps. Lossy compression can introduce additional damage unrelated to resampling.

## 5. One subtle issue: page alignment versus text alignment

I'd also reconsider whether page-edge alignment should dictate the final rotation.

Suppose the detected page edge is at $`0.4^\circ`$, while the text lines are at $`0.1^\circ`$. If your goal is readable text and good OCR, the text angle is likely the more useful target for the final orientation. But if the page boundary is essential for the visual presentation of the book, you might prefer a compromise or preserve the physical page geometry.

The correct choice depends on your output goals:

- **OCR and searchable text:** prioritize text baselines.
- **Visually faithful page scans:** prioritize the page's physical geometry, while avoiding unnecessary corrections.
- **Both:** estimate both, choose a deliberate final geometry, and apply it once.

There is no need to force the text lines and the physical page edges to be perfectly horizontal and vertical if the source itself doesn't satisfy that geometry.

## 6. My recommendation for your repository

For your [`065-remove-page-borders.py`](https://github.com/milahu/hocr-files-template-repo/blob/main/065-remove-page-borders.py) and [`070-deskew.py`](https://github.com/milahu/hocr-files-template-repo/blob/main/070-deskew.py), I'd make the following change:

1.  Keep the page-edge detection and border-removal logic.
2.  Keep the text-based skew estimation.
3.  Change the pipeline so that both steps contribute geometric parameters to a single final transformation.
4.  Apply that transformation to the original image only once.
5.  Compare the output against the current pipeline using a few difficult pages: small text, fine serifs, faint printing, and pages with imperfect edges.

I'd retain `070`'s *functionality* even if its pixel-transforming operation disappears as a separate step.

One final point: your pipeline also has `072-deskew-fix-page-size.py`. Check whether it performs any additional image resampling. If it merely adjusts metadata or output dimensions, that is different from rotating or warping pixels; if it resizes the image, it should be considered in the same quality analysis.

**Bottom line:** merging the two geometric operations is the best default for image quality. Removing text-based deskewing would sacrifice a useful correction to avoid a resampling loss that you can instead eliminate by redesigning the pipeline.

## Response:

11/1/2026, 8:28:00 AM

------------------------------------------------------------------------

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
