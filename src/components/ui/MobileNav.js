"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Bell, Thermometer, Settings, Menu } from "lucide-react";
import { clsx } from "clsx";

export default function MobileNav() {
    const pathname = usePathname();

    const navItems = [
        {
            name: "Dashboard",
            href: "/dashboard",
            icon: Home,
        },
        {
            name: "Alerts",
            href: "/dashboard/alerts",
            icon: Bell,
        },
        {
            name: "Temp",
            href: "/dashboard/temperature",
            icon: Thermometer,
        },
        {
            name: "Settings",
            href: "/settings", // Assuming a settings page exists or will exist
            icon: Settings,
        },
    ];

    return (
        <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#0A1029] border-t border-white/10 md:hidden pb-safe">
            <div className="flex items-center justify-around h-16 px-2">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={clsx(
                                "flex flex-col items-center justify-center w-full h-full space-y-1",
                                isActive ? "text-blue-500" : "text-gray-400 hover:text-gray-200"
                            )}
                        >
                            <Icon size={24} />
                            <span className="text-[10px] font-medium">{item.name}</span>
                        </Link>
                    );
                })}
            </div>
        </div>
    );
}
