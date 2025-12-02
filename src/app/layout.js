import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ToastProvider } from "@/components/ui/ToastContext";
import MobileNav from "@/components/ui/MobileNav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "ELROI Predictive Maintenance",
  description: "Predictive maintenance platform by ELROI",
  manifest: "/manifest.json",
  viewport: "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0",
  themeColor: "#050A24",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-[#050A24] text-[#E6E9F0] pb-16 md:pb-0`}
      >
        <ToastProvider>
          {children}
        </ToastProvider>
        <MobileNav />
      </body>
    </html>
  );
}