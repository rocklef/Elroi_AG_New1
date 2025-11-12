import Image from 'next/image'

export default function Layer2Left() {
  return (
    <div className="w-1/2 bg-white flex flex-col justify-center items-center p-12 border border-[#1e3a8a]/30 rounded-3xl shadow-2xl">
      <div className="text-center max-w-md">
        <div className="mb-6 flex justify-center">
          <Image
            src="/logo.png"
            alt="ELROI Automation Logo"
            width={240}
            height={60}
            priority
            className="object-contain"
          />
        </div>
        <h2 className="text-[#050A24] font-semibold text-3xl mb-6 leading-tight">
          Predictive Maintenance
        </h2>
        <p className="text-gray-700 text-lg font-medium leading-relaxed">
          Advanced analytics for industrial equipment reliability and performance optimization
        </p>
      </div>
    </div>
  )
}