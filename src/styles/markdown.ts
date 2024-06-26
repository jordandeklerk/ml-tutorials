import React from "react"
import styled from "styled-components"

import type typography from "./typography"

const Markdown = styled.article<{ rhythm: (typeof typography)["rhythm"] }>`
  h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    font-weight: var(--font-weight-bold);
  }

  table {
    margin-bottom: var(--sizing-base);
  }
  td,
  th {
    padding: var(--padding-xs);
    border: 1px solid var(--color-gray-3);
  }
  th {
    font-weight: var(--font-weight-semi-bold);
  }

  strong {
    font-weight: var(--font-weight-semi-bold);
  }

  a,
  p {
    font-weight: var(--font-weight-regular);
  }

  a {
    text-decoration: none;
    color: var(--color-blue) !important;
    * {
      color: var(--color-blue) !important;
    }
    &:hover,
    &:active {
      text-decoration: underline;
    }
  }

  & > *:first-child {
    margin-top: 0;
  }

  h1 {
    font-size: 2.5rem;

    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 2rem;
    }
  }

  h2 {
    font-size: 1.75rem;
    line-height: 1.3;
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-top: ${({ rhythm }) => rhythm(2.25)};

    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 1.3125rem;
    }
  }

  h3 {
    font-size: 1.31951rem;
    line-height: 1.3;
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-top: ${({ rhythm }) => rhythm(1.5)};

    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 1.1875rem;
    }
  }

  h4,
  h5,
  h6 {
    margin-bottom: ${({ rhythm }) => rhythm(0.5)};
    margin-top: ${({ rhythm }) => rhythm(1)};
  }

  ul,
  ol {
    margin-top: ${({ rhythm }) => rhythm(1)};
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-left: ${({ rhythm }) => rhythm(1.25)};
  }

  ol {
    list-style: auto;
  }
  ul {
    list-style: disc;
  }

  li > ul,
  li > ol {
    margin-top: 0;
    margin-bottom: 0;
  }

  li > p {
    margin-bottom: 0;
  }

  li > ol,
  li > ul {
    margin-left: ${({ rhythm }) => rhythm(1.25)};
  }

  li {
    margin-bottom: ${({ rhythm }) => rhythm(0.3)};
  }

  li {
    line-height: 1.68;
  }

  p,
  li,
  blockquote {
    font-size: 1.0625rem;
  }

  p {
    line-height: 1.68;
    text-align: left;
    margin-bottom: var(--sizing-md);
  }

  hr {
    margin: var(--sizing-lg) 0;
    background: var(--color-gray-3);
  }

  blockquote {
    border-left: 0.25rem solid var(--color-gray-2);
    padding-left: var(--sizing-base);
    margin: var(--sizing-md) 0;
    * {
      color: var(--color-gray-6);
    }
  }

  img {
    display: block;
  }

  pre,
  code {
    font-family:
      SFMono-Regular,
      Consolas,
      Liberation Mono,
      Menlo,
      monospace;
    background-color: var(--color-code-block);
    font-size: 0.9em; // Adjust the font size here
  }

  pre {
    position: relative;
    border: 1px solid var(--color-gray-3);
    overflow: auto; // Ensure the code block can scroll
  }

  .code-block-wrapper {
    position: relative;
  }

  pre.grvsc-container {
    margin: var(--sizing-md) 0;
  }

  .grvsc-line-highlighted::before {
    background-color: var(--color-code-highlight) !important;
    box-shadow: inset 4px 0 0 0 var(--color-code-highlight-border) !important;
  }

  *:not(pre) > code {
    background-color: var(--color-code);
    padding: 0.2rem 0.4rem;
    margin: 0;
    font-size: 85%;
    border-radius: 3px;
  }

  .copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    font-size: 12px;
    background-color: var(--color-code-button);
    color: var(--color-code-button-text);
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }

  .copy-button.copied {
    background-color: var(--color-code-button-copied);
  }

  // Styles for details (proofs)
  details {
    background-color: var(--color-code-block);
    border: 1px solid var(--color-gray-3);
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  summary {
    cursor: pointer;
    font-weight: var(--font-weight-bold);
    margin-bottom: 0.5rem;
  }

  details[open] summary {
    margin-bottom: 1rem;
  }

  // Updated styles for theorems
  .theorem {
    background-color: var(--color-code-block);
    border: 1px solid var(--color-gray-3);
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .theorem .theorem-title,
  .theorem p.theorem-title {
    font-weight: var(--font-weight-bold);
    margin-bottom: 0.5rem;
    color: #c85417 !important;  // New color for theorem title with !important
  }

  .theorem-content {
    margin-top: 0.5rem;
    color: #3a8ab0;
  }

  // Override styles for paragraphs, math elements, and list items within theorems
  .theorem p,
  .theorem .katex,
  .theorem .katex-html,
  .theorem ul,
  .theorem ol,
  .theorem li {
    color: #3a8ab0;
    margin-bottom: 0;
  }

  // Ensure inline math is colored correctly
  .theorem .katex-html * {
    color: #3a8ab0 !important;
  }

  // For MathJax (if you're using it instead of KaTeX)
  .theorem .MathJax {
    color: #3a8ab0 !important;
  }

  // Style list items within theorems
  .theorem ul,
  .theorem ol {
    margin-top: ${({ rhythm }) => rhythm(0.5)};
    margin-bottom: ${({ rhythm }) => rhythm(0.5)};
    padding-left: ${({ rhythm }) => rhythm(0)};
  }

  .theorem li {
    color: #3a8ab0;
    margin-bottom: ${({ rhythm }) => rhythm(0.3)};
  }

  .theorem li::marker {
    color: #3a8ab0;
  }

  // Ensure code blocks within details are not styled as code blocks within the details box
  details pre,
  details code {
    background-color: transparent;
    border: none;
    padding: 0;
    font-size: inherit;
  }

  // Explicitly set the font size for code blocks within details to match regular code blocks
  details pre,
  details code {
    font-size: 0.9em; // Match the font size of regular code blocks
  }
`

export default Markdown