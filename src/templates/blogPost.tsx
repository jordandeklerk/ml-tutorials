import React, { useCallback, useEffect } from "react"
import { type PageProps, graphql } from "gatsby"
import styled from "styled-components"

import Comment from "~/src/components/comment"
import SEO from "~/src/components/seo"
import Layout from "~/src/layouts/layout"
import Category from "~/src/styles/category"
import DateTime from "~/src/styles/dateTime"
import Markdown from "~/src/styles/markdown"
import { rhythm } from "~/src/styles/typography"

const BlogPost: React.FC<PageProps<Queries.Query>> = ({ data }) => {
  const { markdownRemark } = data
  const { frontmatter, html } = markdownRemark!
  const { title, desc, thumbnail, date, category } = frontmatter!

  const ogImagePath =
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    thumbnail &&
    thumbnail?.childImageSharp?.gatsbyImageData!.images!.fallback!.src

  const copyToClipboard = useCallback((text: string) => {
    navigator.clipboard.writeText(text)
  }, [])

  const handleCopyClick = useCallback((event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault()
    const button = event.target as HTMLButtonElement
    const pre = button.parentElement
    const code = pre?.querySelector("code")?.innerText
    if (code) {
      copyToClipboard(code)
      button.innerText = "Copied!"
      button.classList.add("copied")
      setTimeout(() => {
        button.innerText = "Copy"
        button.classList.remove("copied")
      }, 2000)
    }
  }, [copyToClipboard])

  useEffect(() => {
    const codeBlocks = document.querySelectorAll("pre")
    codeBlocks.forEach(pre => {
      const wrapper = document.createElement("div")
      wrapper.classList.add("code-block-wrapper")
      pre.parentNode?.insertBefore(wrapper, pre)
      wrapper.appendChild(pre)

      const button = document.createElement("button")
      button.innerText = "Copy"
      button.classList.add("copy-button")
      wrapper.appendChild(button)
    })
  }, [])

  return (
    <Layout>
      <SEO title={title} desc={desc} image={ogImagePath} />
      <main>
        <article>
          <OuterWrapper>
            <InnerWrapper>
              <div>
                <header>
                  <Info>
                    <PostCategory>{category}</PostCategory>
                    <Time dateTime={date!}>{date}</Time>
                  </Info>
                  <Title>{title}</Title>
                  <Desc>{desc}</Desc>
                </header>
                <Divider />
                <Markdown
                  dangerouslySetInnerHTML={{ __html: html ?? "" }}
                  rhythm={rhythm}
                  onClick={event => {
                    if (event.target instanceof HTMLButtonElement && event.target.classList.contains("copy-button")) {
                      handleCopyClick(event)
                    }
                  }}
                />
              </div>
            </InnerWrapper>
          </OuterWrapper>
        </article>
        <CommentWrap>
          <Comment />
        </CommentWrap>
      </main>
    </Layout>
  )
}

const OuterWrapper = styled.div`
  margin-top: var(--sizing-xl);

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    margin-top: var(--sizing-lg);
  }
`

const InnerWrapper = styled.div`
  width: 100%;
  max-width: 52%;
  margin: 0 auto;
  padding-bottom: var(--sizing-lg);

  @media (max-width: ${({ theme }) => theme.device.md}) {
    max-width: 75%;
  }

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    max-width: 87.5%;
  }
`

const CommentWrap = styled.section`
  width: 100%;
  padding: 0 var(--padding-sm);
  margin: 0 auto;
  margin-bottom: var(--sizing-xl);

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    width: auto;
  }
`

const PostCategory = styled(Category)`
  font-size: 0.875rem;
  font-weight: var(--font-weight-semi-bold);
`

const Info = styled.div`
  margin-bottom: var(--sizing-md);
`

const Time = styled(DateTime)`
  display: block;
  margin-top: var(--sizing-xs);
`

const Desc = styled.p`
  margin-top: var(--sizing-lg);
  line-height: 1.5;
  font-size: var(--text-lg);

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    line-height: 1.31579;
    font-size: 1.1875rem;
  }
`

const Divider = styled.div`
  width: 100%;
  height: 1px;
  background-color: var(--color-gray-3);
  margin-top: var(--sizing-lg);
  margin-bottom: var(--sizing-lg);
`

const Title = styled.h1`
  font-weight: var(--font-weight-bold);
  line-height: 1.1875;
  font-size: var(--text-xl);

  @media (max-width: ${({ theme }) => theme.device.md}) {
    line-height: 1.21875;
    font-size: 2.5rem;
  }

  @media (max-width: ${({ theme }) => theme.device.sm}) {
    line-height: 1.21875;
    font-size: 2rem;
  }
`

export const query = graphql`
  query BlogPostPage($slug: String!) {
    markdownRemark(fields: { slug: { eq: $slug } }) {
      html
      frontmatter {
        title
        desc
        thumbnail {
          childImageSharp {
            gatsbyImageData(placeholder: BLURRED, layout: FIXED)
          }
        }
        date(formatString: "YYYY-MM-DD")
        category
      }
    }
  }
`

export default BlogPost
